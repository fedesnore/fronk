/* test_config.c - configuration grammar, bounds, invalid fields and atomic load. */
#include "test.h"
#include "../src/config.h"
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unistd.h>

static char error[256];
static int load_bytes(const char *text, size_t len, fr_config *out) {
    char path[]="/tmp/fronk-config.XXXXXX";
    int fd=mkstemp(path);
    if (fd<0) return -2;
    int written=write(fd,text,len)==(ssize_t)len;
    if (close(fd)<0) written=0;
    int rc=written ? config_load(path,out,error,sizeof(error)) : -2;
    if (unlink(path)<0) return -2;
    return rc;
}
static int load(const char *text, fr_config *out) { return load_bytes(text,strlen(text),out); }
int main(void) {
    test_begin("config"); fr_config cfg;
    config_defaults(&cfg);
    check(!strcmp(cfg.bind,"127.0.0.1") && cfg.control_port==7700, "loopback defaults");
    char hostname[256]; int named=!gethostname(hostname,sizeof(hostname)); char *dot=named?strchr(hostname,'.'):NULL; if (dot) *dot=0;
    check(named && !strcmp(cfg.name,hostname),"name defaults to short hostname");
    check(cfg.mem_margin==FR_DEFAULT_MARGIN && cfg.head_reserve==UINT64_C(10000000000),"platform margin sentinel and 10 decimal GB head reserve");
    check(!load("mem-margin-gb 1.5\nhead-reserve-gb 0\nstate-dir /tmp/fronk-state\nprofile-file /tmp/profile.txt\n",&cfg) && cfg.mem_margin==1500000000 && !cfg.head_reserve && !strcmp(cfg.state_dir,"/tmp/fronk-state"),"exact decimal GB and isolated paths");
    check(!cfg.backend[0] && !load("backend avx2\n",&cfg) && !strcmp(cfg.backend,"avx2") && !load("backend auto\n",&cfg) && !strcmp(cfg.backend,"auto") &&
          load("backend AVX2\n",&cfg)<0 && load("backend avx2\nbackend neon\n",&cfg)<0,"backend names one built-in or own backend; auto races them");
    check(!cfg.head_blocks && !load("head-blocks 4\n",&cfg) && cfg.head_blocks==4 && load("head-blocks 21\n",&cfg)<0 && load("head-blocks x\n",&cfg)<0,"head-blocks fixes 0..20 head blocks");
    check(load("mem-margin-gb 1e3\n",&cfg)<0 && load("mem-margin-gb -1\n",&cfg)<0 && load("head-reserve-gb 18446744074\n",&cfg)<0,"invalid or overflowing GB rejected");
    check(!load("mem-margin-gb 0.000000001\n",&cfg) && cfg.mem_margin==1 && load("mem-margin-gb 0.0000000001\n",&cfg)<0,"byte precision is exact and bounded");
    check(load("# comment\n name test-node\ncontrol-port 0 # ephemeral\ndata-port 7703\npassword local-secret",&cfg)==0 &&
          !strcmp(cfg.name,"test-node") && cfg.control_port==0 && !strcmp(cfg.password,"local-secret"), "comments, final line and ephemeral port");
    fr_config saved=cfg;
    check(load("unknown secret-value\n",&cfg)==-1 && !strstr(error,"secret-value"), "unknown keys without leaking values");
    check(!memcmp(&saved,&cfg,sizeof(cfg)), "failed load leaves config unchanged");
    check(load("name one\nname two\n",&cfg)==-1, "duplicate keys rejected");
    check(load("name\n",&cfg)==-1 && load("name one two\n",&cfg)==-1, "missing and extra values rejected");
    check(load("control-port -1\n",&cfg)==-1 && load("control-port 65536\n",&cfg)==-1, "port bounds");
    check(load("control-port 1x\n",&cfg)==-1 && load("control-port +12\n",&cfg)==-1, "strict decimal ports");
    check(load("data-port 0\ndiscovery-port 0\n",&cfg)==0 && !cfg.data_port && !cfg.discovery_port, "ephemeral node UDP ports");
    check(load("control-port 7701\n",&cfg)==-1, "control and data ports differ");
    check(load("bind hostname\n",&cfg)==-1, "bind must be IPv4");
    check(load("multicast 127.0.0.1\n",&cfg)==-1, "multicast address must be multicast");
    check(load("name invalid/name\n",&cfg)==-1, "safe node names");
    char long_line[600]; memset(long_line,'x',sizeof(long_line));
    check(load_bytes(long_line,sizeof(long_line),&cfg)==-1, "oversized line");
    check(load_bytes("name a\0hidden\n",14,&cfg)==-1, "embedded NUL rejected");
    char long_name[80]; memset(long_name,'x',sizeof(long_name)); memcpy(long_name,"name ",5);
    check(load_bytes(long_name,sizeof(long_name),&cfg)==-1, "field length bounded");
    check(config_load("/nonexistent/fronk.conf",&cfg,error,sizeof(error))==-1, "missing file");
    check(load("",&cfg)==0 && cfg.control_port==7700, "empty config uses defaults");
    check(!strcmp(cfg.interface,"127.0.0.1") && cfg.discovery_port==7702,"local discovery defaults");
    check(load("interface 192.0.2.2\nbroadcast 192.0.2.255\ndiscovery-port 8800\n",&cfg)==0 && cfg.discovery_port==8800,"explicit LAN interface and broadcast");
    check(load("interface 0.0.0.0\n",&cfg)<0 && load("interface 239.7.7.7\n",&cfg)<0,"interface must be a concrete unicast address");
    check(load("discovery-port 7701\n",&cfg)<0,"shared discovery and data UDP ports must differ");
    check(!load("interfaces auto\n",&cfg) && !strcmp(cfg.interfaces,"auto"),"automatic interface inventory is explicit");
    check(!load("interfaces 127.0.0.1,192.0.2.1\n",&cfg) && !strcmp(cfg.interfaces,"127.0.0.1,192.0.2.1"),"bounded local IPv4 interface list");
    check(load("interfaces 127.0.0.1,\n",&cfg)<0 && load("interfaces ,127.0.0.1\n",&cfg)<0 &&
          load("interfaces 0.0.0.0\n",&cfg)<0 && load("interfaces localhost\n",&cfg)<0,"invalid interface lists refused without OS changes");
    config_defaults(&cfg); check(cfg.burst==1,"default measured selection burst is one");
    check(!load("burst 4\n",&cfg) && cfg.burst==4,"configured BENCH burst");
    check(load("burst 0\n",&cfg)<0 && load("burst 5\n",&cfg)<0,"benchmark burst outside frozen 1..4 range rejected");
    check(!load("name head\nbind 0.0.0.0\ninterfaces 198.51.100.1,192.0.2.1\nnode m1pro@198.51.100.2\nnode imac@192.0.2.4\nnode shuttle@192.0.2.3\n",&cfg) && cfg.peer_count==3 && !strcmp(cfg.peers[2].name,"shuttle"),"real cluster roster keeps the key/value grammar");
    check(load("node a@10.1.1.1\nnode a@10.1.1.2\n",&cfg)<0 && load("node a@10.1.1.1\nnode b@10.1.1.1\n",&cfg)<0,"duplicate node name or address refused");
    check(load("node a@239.1.1.1\n",&cfg)<0 && load("node bad/name@10.1.1.1\n",&cfg)<0,"invalid roster entry refused");
    check(!setenv("FRONK_TEST_SECRET","synthetic-test-secret",1) && !load("password-env FRONK_TEST_SECRET\n",&cfg) && !strcmp(cfg.password,"synthetic-test-secret"),"runtime secret read without putting it in config");
    check(load("password x\npassword-env FRONK_TEST_SECRET\n",&cfg)<0 && !strstr(error,"synthetic-test-secret"),"ambiguous secret source rejected without disclosure");
    unsetenv("FRONK_TEST_SECRET");
    check(load("password-env FRONK_TEST_SECRET\n",&cfg)<0,"missing runtime secret fails explicitly");
    char dir[]="/tmp/fronk-config-XXXXXX"; check(mkdtemp(dir)!=NULL,"temporary state dir");
    char conf[128]; snprintf(conf,sizeof(conf),"state-dir %s\npassword-env FRONK_TEST_SECRET\n",dir);
    check(load(conf,&cfg)<0,"no saved cluster secret and no environment fails");
    setenv("FRONK_TEST_SECRET","stable-cluster-secret",1);
    check(!load(conf,&cfg) && !strcmp(cfg.password,"stable-cluster-secret"),"environment secret saved to state-dir");
    unsetenv("FRONK_TEST_SECRET");
    check(!load(conf,&cfg) && !strcmp(cfg.password,"stable-cluster-secret"),"saved cluster secret reused across restarts");
    char file[160]; snprintf(file,sizeof(file),"%s/cluster.secret",dir); struct stat st;
    check(!stat(file,&st) && (st.st_mode&0777)==0600,"cluster secret file is owner-only");
    unlink(file); rmdir(dir);
    return test_end();
}
