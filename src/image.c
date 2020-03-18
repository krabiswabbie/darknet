#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>

int windows = 0;

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

void normalize_image(image p)
{
    int i;
    float min = 9999999;
    float max = -999999;

    for(i = 0; i < p.h*p.w*p.c; ++i){
        float v = p.data[i];
        if(v < min) min = v;
        if(v > max) max = v;
    }
    if(max - min < .000000001){
        min = 0;
        max = 1;
    }
    for(i = 0; i < p.c*p.w*p.h; ++i){
        p.data[i] = (p.data[i] - min)/(max-min);
    }
}

image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

image float_to_image(int w, int h, int c, float *data)
{
    image out = make_empty_image(w,h,c);
    out.data = data;
    return out;
}

void scale_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}

image get_image_layer(image m, int l)
{
    image out = make_image(m.w, m.h, 1);
    int i;
    for(i = 0; i < m.h*m.w; ++i){
        out.data[i] = m.data[i+l*m.h*m.w];
    }
    return out;
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
