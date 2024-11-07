#include <stdlib.h>
#include <ctype.h>
#include "b64.h"


int b64_buf_malloc(b64_buffer_t * buf)
{
	buf->ptr = (char *)malloc(B64_BUFFER_SIZE);
	if(!buf->ptr) return -1;

	buf->bufc = 1;

	return 0;
}

int b64_buf_realloc(b64_buffer_t* buf, size_t size)
{
	if (size > buf->bufc * B64_BUFFER_SIZE)
	{
		while (size > buf->bufc * B64_BUFFER_SIZE) buf->bufc++;
		buf->ptr = (char *)realloc(buf->ptr, B64_BUFFER_SIZE * buf->bufc);
		if (!buf->ptr) return -1;
	}

	return 0;
}