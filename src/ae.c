/* ae.c - kqueue/epoll readiness for the single control-plane event loop.
 *
 * Level triggering makes partial reads/writes safe. Opaque generation tags
 * survive the kernel round trip so a recycled connection cannot inherit an
 * old event. A fixed stack batch avoids allocation during polling.
 */
#include "ae.h"
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/event.h>
#include <time.h>
#else
#include <sys/epoll.h>
#endif

const char *ae_name(void) {
#ifdef __APPLE__
    return "kqueue";
#else
    return "epoll";
#endif
}

int ae_open(void) {
#ifdef __APPLE__
    int fd = kqueue();
    if (fd >= 0 && fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
        int error = errno; close(fd); errno = error; return -1;
    }
    return fd;
#else
    return epoll_create1(EPOLL_CLOEXEC);
#endif
}

int ae_set(int loop, int fd, int old_mask, int new_mask, uint64_t tag) {
    if (fd < 0 || ((old_mask | new_mask) & ~(AE_READ | AE_WRITE))) {
        errno = EINVAL; return -1;
    }
#ifdef __APPLE__
    _Static_assert(sizeof(uintptr_t) == sizeof(uint64_t), "64-bit target required");
    for (int bit = AE_READ; bit <= AE_WRITE; bit <<= 1) {
        if (!((old_mask | new_mask) & bit)) continue;
        struct kevent change;
        int add = new_mask & bit;
        EV_SET(&change, (uintptr_t)fd, bit == AE_READ ? EVFILT_READ : EVFILT_WRITE,
               add ? EV_ADD | EV_ENABLE : EV_DELETE, 0, 0, (void *)(uintptr_t)tag);
        if (kevent(loop, &change, 1, NULL, 0, NULL) < 0) {
            if (!add && errno == ENOENT) continue;
            return -1;
        }
    }
    return 0;
#else
    if (!(old_mask | new_mask)) return 0;
    struct epoll_event event = {.data.u64=tag};
    if (new_mask & AE_READ) event.events |= EPOLLIN;
    if (new_mask & AE_WRITE) event.events |= EPOLLOUT;
    int op = !new_mask ? EPOLL_CTL_DEL : old_mask ? EPOLL_CTL_MOD : EPOLL_CTL_ADD;
    return epoll_ctl(loop, op, fd, &event);
#endif
}

int ae_wait(int loop, ae_event *events, int cap, int timeout_ms) {
    if (!events || cap < 1 || cap > AE_MAX_EVENTS || timeout_ms < -1) {
        errno = EINVAL; return -1;
    }
#ifdef __APPLE__
    struct kevent ready[AE_MAX_EVENTS];
    struct timespec timeout = {timeout_ms / 1000, (timeout_ms % 1000) * 1000000L};
    int n = kevent(loop, NULL, 0, ready, cap, timeout_ms < 0 ? NULL : &timeout);
    for (int i = 0; i < n; ++i) {
        events[i].tag = (uint64_t)(uintptr_t)ready[i].udata;
        events[i].mask = ready[i].filter == EVFILT_READ ? AE_READ : AE_WRITE;
        if (ready[i].flags & (EV_EOF | EV_ERROR)) events[i].mask |= AE_READ | AE_WRITE;
    }
#else
    struct epoll_event ready[AE_MAX_EVENTS];
    int n = epoll_wait(loop, ready, cap, timeout_ms);
    for (int i = 0; i < n; ++i) {
        events[i].tag = ready[i].data.u64;
        events[i].mask = 0;
        if (ready[i].events & EPOLLIN) events[i].mask |= AE_READ;
        if (ready[i].events & EPOLLOUT) events[i].mask |= AE_WRITE;
        if (ready[i].events & (EPOLLERR | EPOLLHUP)) events[i].mask |= AE_READ | AE_WRITE;
    }
#endif
    return n < 0 && errno == EINTR ? 0 : n;
}
