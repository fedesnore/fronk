/* test_hw_vm.c - older macOS may return a shorter VM structure than the SDK.
 * Accept replies containing all fields we consume; reject truncated prefixes.
 */
#include "test.h"
#include "../src/hw.h"
#include <mach/mach.h>
#include <stddef.h>
#include <string.h>
static unsigned returned_count=38;
kern_return_t fr_test_host_statistics64(host_t host, host_flavor_t flavor, host_info64_t out, mach_msg_type_number_t *count) {
    (void)host; (void)flavor;
    vm_statistics64_data_t vm={0};
    vm.free_count=100; vm.inactive_count=200; vm.purgeable_count=300;
    memcpy(out,&vm,sizeof(vm)); *count=returned_count; return KERN_SUCCESS;
}
int main(void) {
    test_begin("hw_vm"); fr_hwinfo h={0};
    check(!hw_probe(&h) && h.available>0,"macOS 15 VM reply accepted with a newer SDK");
    returned_count=(unsigned)((offsetof(vm_statistics64_data_t,purgeable_count)+sizeof(((vm_statistics64_data_t *)0)->purgeable_count))/sizeof(integer_t))-1;
    h.total=123;
    check(hw_probe(&h)<0 && h.total==123,"VM reply missing required fields fails without replacing output");
    return test_end();
}
