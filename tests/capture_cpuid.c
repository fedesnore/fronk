/* capture_cpuid.c - read-only register evidence for x86 decoder fixtures.
 * Compile on an authorized x86 node; never infer ISA from sysctl/proc strings.
 */
#include <cpuid.h>
#include <stdio.h>
int main(void) {
    unsigned a,b,c,d,one=0,seven=0,low=0,high=0;
    if (__get_cpuid(1,&a,&b,&c,&d)) one=c;
    if (__get_cpuid_count(7,0,&a,&b,&c,&d)) seven=b;
    if ((one&0x0c000000)==0x0c000000) __asm__ volatile("xgetbv":"=a"(low),"=d"(high):"c"(0));
    printf("leaf1_ecx=0x%08x leaf7_ebx=0x%08x xcr0=0x%08x%08x\n",one,seven,high,low);
    return 0;
}
