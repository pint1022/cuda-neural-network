#ifndef __READER_API_H__
#define __READER_API_H__
#define KB(x) ((x)*1024L)

char* read_numpy(char * file_name, int length,  int *row, int *col);
unsigned int reverse_int(unsigned int i);
__global__ void g_reverse_int(unsigned int i);

#endif