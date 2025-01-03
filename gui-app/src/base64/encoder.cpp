
/**
 * `encode.c' - b64
 *
 * copyright (c) 2014 joseph werle
 */

#include <stdio.h>
#include <stdlib.h>
#include "b64.h"


char *
b64_encode (const unsigned char *src, size_t len) {
  int i = 0;
  int j = 0;
  b64_buffer_t encbuf;
  size_t size = 0;
  unsigned char buf[4];
  unsigned char tmp[3];

  // alloc
  if(b64_buf_malloc(&encbuf) == -1) { return NULL; }

  // parse until end of source
  while (len--) {
    // read up to 3 bytes at a time into `tmp'
    tmp[i++] = *(src++);

    // if 3 bytes read then encode into `buf'
    if (3 == i) {
      buf[0] = (tmp[0] & 0xfc) >> 2;
      buf[1] = ((tmp[0] & 0x03) << 4) + ((tmp[1] & 0xf0) >> 4);
      buf[2] = ((tmp[1] & 0x0f) << 2) + ((tmp[2] & 0xc0) >> 6);
      buf[3] = tmp[2] & 0x3f;

      // allocate 4 new byts for `enc` and
      // then translate each encoded buffer
      // part by index from the base 64 index table
      // into `encbuf.ptr' unsigned char array
      if (b64_buf_realloc(&encbuf, size + 4) == -1) return NULL;

      for (i = 0; i < 4; ++i) {
        encbuf.ptr[size++] = b64_table[buf[i]];
      }

      // reset index
      i = 0;
    }
  }

  // remainder
  if (i > 0) {
    // fill `tmp' with `\0' at most 3 times
    for (j = i; j < 3; ++j) {
      tmp[j] = '\0';
    }

    // perform same codec as above
    buf[0] = (tmp[0] & 0xfc) >> 2;
    buf[1] = ((tmp[0] & 0x03) << 4) + ((tmp[1] & 0xf0) >> 4);
    buf[2] = ((tmp[1] & 0x0f) << 2) + ((tmp[2] & 0xc0) >> 6);
    buf[3] = tmp[2] & 0x3f;

    // perform same write to `encbuf->ptr` with new allocation
    for (j = 0; (j < i + 1); ++j) {
      if (b64_buf_realloc(&encbuf, size + 1) == -1) return NULL;

      encbuf.ptr[size++] = b64_table[buf[j]];
    }

    // while there is still a remainder
    // append `=' to `encbuf.ptr'
    while ((i++ < 3)) {
      if (b64_buf_realloc(&encbuf, size + 1) == -1) return NULL;

      encbuf.ptr[size++] = '=';
    }
  }

  // Make sure we have enough space to add '\0' character at end.
  if (b64_buf_realloc(&encbuf, size + 1) == -1) return NULL;
  encbuf.ptr[size] = '\0';

  return encbuf.ptr;
}