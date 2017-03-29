#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <omp.h>


//SSE STUFF
#ifdef __SSE__
#include <xmmintrin.h>
#endif

//const vars
const int XPICR = 1000;
const int YPICR = 1000;


uint IterateMandelbrot(float a, float b );//prototype
void GenerateMandleBlock(uint* buffer, int yStart, int yEnd);
int getYstep(int cahcesize, int yStart, int yEnd);
void GenMandleSSEOMP(uint* buffer, int yStart, int yEnd);
__m128i IterateMandelbrotOMPOPT( __m128 a, __m128 b );

uint* GenerateMandleCache(int yStart, int yEnd, int cachesize, int localbuffersize){
	int yStep = getYstep(cachesize,yStart,yEnd);
	uint* localbuffer = (uint*)malloc(sizeof(uint)*localbuffersize+1);
	
	
	int i, j;
	int iCnt = 0;
	
	
	for(i = yStart ; i<=yEnd; i+=yStep){
		j = i + yStep;
		if(j>yEnd)
			j = yEnd;//safely bound j
			
		int blockSize = (int)(j - i)*XPICR;
		uint* iterBuffer = (uint*)malloc(sizeof(uint)*blockSize+1);
		GenerateMandleBlock(iterBuffer, i, j);
		memcpy(localbuffer+(iCnt),     iterBuffer, blockSize * sizeof(uint));
		iCnt+=blockSize;
		
	}
	return localbuffer;
	
}

int getYstep(int cahcesize, int yStart, int yEnd){
	int test = (((XPICR/1000)*4));
	if (test <= 0)
		test = 1;
	int yStep = (int)((cahcesize)/test);//defines y step
	if (yStep<1)
		yStep = 1;
	if(yStep>cahcesize)
		yStep = cahcesize;
	return yStep;
}

void GenerateMandleBlock(uint* buffer, int yStart, int yEnd){
	omp_set_num_threads(omp_get_max_threads());
	int i,j;
	int iCnt = 0;
	float IXRES = 1.0f/(float)XPICR;
	float IYRES = 1.0f/(float)YPICR;
	float a;
	float b;
	
	#pragma omp parallel for private(j, i) schedule(runtime) ordered
     for(j=yStart; j < yEnd; j++ )
        for(i=0; i < XPICR; i++ ){
            a = -2.05f + 3.00f*(float)i*IXRES;
            b =  1.10f - 2.24f*(float)j*IYRES;
            
           #pragma omp ordered
			{
				buffer[iCnt++] = IterateMandelbrot(a,b);
			}
           
          
        }
}

void GenerateMandle(uint* buffer, int yStart, int yEnd){
	
	int i,j;
	int iCnt = 0;
	float IXRES = 1.0f/(float)XPICR;
	float IYRES = 1.0f/(float)YPICR;
	
     for(j=yStart; j < yEnd; j++ )
        for(i=0; i < XPICR; i++ ){
            const float a = -2.05f + 3.00f*(float)i*IXRES;
            const float b =  1.10f - 2.24f*(float)j*IYRES;
            
            *(buffer++) = IterateMandelbrot(a,b);
        }
}


uint IterateMandelbrot(float a, float b ){
    float x, y, x2, y2;
   
    x = x2 = 0.0f;
    y = y2 = 0.0f;
    float m2;
    // escape time algorithm
    int i;
    float h = 0.0;
    volatile int flag = 0;
    
 
    for(i=0; i< 2048; i++ )
    {
		if (flag==1)
			continue;
        y = 2.0f*x*y+b;
        x = x2-y2+a;

        x2 = x*x;
        y2 = y*y;
	
        m2 = x2+y2;
        if( m2>100.0f )
            flag = 1;
        h+=1.0;
    }
    h=h+1.0-log2(.5*log(m2));
    return (h/100.0)*256;
}

int PPMmin(int i1, int i2){
  if (i1 < i2)
    return i1;
  else
    return i2;
}

//PPM SAVER
void MandleSavePPM(uint* MandleBuffer){
	//ASCII PPM P3
	FILE *fsave = fopen("result.ppm", "wt");
	int i, j, k, jmax;
	int index = 0;
	
	fprintf (fsave, "P3\n");
	fprintf (fsave, "%d  %d\n", XPICR, YPICR);
	fprintf (fsave, "%d\n", 255);

	for (i = 0; i<YPICR; i++){
		for (k=0; k<XPICR; k=k+4){
			jmax = PPMmin(k + 4, XPICR);
			for (j = k; j < jmax; j++){
				index = ((j) * XPICR) + i;
				fprintf (fsave, "  %d  %d  %d", MandleBuffer[index]%256 , MandleBuffer[index]%256 , MandleBuffer[index]%256);
					
			}
			fprintf(fsave, "\n" );
    }
    
  }
  fclose (fsave);
}



///SSE
uint* GenMandleSSEOMPCache(int yStart, int yEnd, int cachesize, int localbuffersize){
	int yStep = getYstep(cachesize,yStart,yEnd);
	uint* localbuffer = (uint*)malloc(sizeof(uint)*localbuffersize+1);
	
	
	int i, j;
	int iCnt = 0;
	
	
	for(i = yStart ; i<=yEnd; i+=yStep){
		j = i + yStep;
		if(j>yEnd)
			j = yEnd;//safely bound j
			
		int blockSize = (int)(j - i)*XPICR;
		uint* iterBuffer = (uint*)malloc(sizeof(uint)*blockSize+1);
		GenMandleSSEOMP(iterBuffer, i, j);
		memcpy(localbuffer+(iCnt),     iterBuffer, blockSize * sizeof(uint));
		iCnt+=blockSize;
		
	}
	return localbuffer;
	
}


void GenMandleSSEOMP(uint* buffer, int yStart, int yEnd){
    __m128i  *buffer4 = (__m128i *)buffer;
    const __m128 ixres = _mm_set1_ps( 1.0f/(float)XPICR );
    const __m128 iyres = _mm_set1_ps( 1.0f/(float)YPICR );

    int j=0,i=0;
    int iCnt = 0;
   // #pragma omp parallel for schedule(runtime) private(i, j) ordered
    for(j=yStart; j < yEnd; j++ )
    for(i=0; i < XPICR; i+=4 )
    {
         __m128  a, b;

        a = _mm_set_ps( i+3, i+2, i+1, i+0 );
        a = _mm_mul_ps( a, ixres );
        a = _mm_mul_ps( a, _mm_set1_ps( 3.00f) );
        a = _mm_add_ps( a, _mm_set1_ps(-2.25f) );

        b = _mm_set1_ps( (float)j );
        b = _mm_mul_ps( b, iyres );
        b = _mm_mul_ps( b, _mm_set1_ps(-2.24f) );
        b = _mm_add_ps( b, _mm_set1_ps( 1.12f) );


		// #pragma omp ordered
			{
			_mm_store_si128( buffer4++, IterateMandelbrotOMPOPT( a, b ));
			}


    }
}
inline __m128i IterateMandelbrotOMPOPT( __m128 a, __m128 b ){
    __m128  x, y, x2, y2, m2;
    __m128  co, ite;

    unsigned int i;

    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 th  = _mm_set1_ps(4.0f);


        x   = _mm_setzero_ps();
        y   = _mm_setzero_ps();
        x2  = _mm_setzero_ps();
        y2  = _mm_setzero_ps();
        co  = _mm_setzero_ps();
        ite = _mm_setzero_ps();

    //volatile int flag=0;

    for( i=0; i < 2048; i++ ){
        y  = _mm_mul_ps( x, y );
        y  = _mm_add_ps( _mm_add_ps(y,y),   b );
        x  = _mm_add_ps( _mm_sub_ps(x2,y2), a );

        x2 = _mm_mul_ps( x, x );
        y2 = _mm_mul_ps( y, y );

        m2 = _mm_add_ps(x2,y2);
        co = _mm_or_ps( co, _mm_cmpgt_ps( m2, th ) );


        ite = _mm_add_ps( ite, _mm_andnot_ps( co, one ) );
        if( _mm_movemask_ps( co )==0x0f )
           break;
        }

    // create color
    const __m128i bb = _mm_cvtps_epi32( ite );
    const __m128i gg = _mm_slli_si128( bb, 1 );
    const __m128i rr = _mm_slli_si128( bb, 2 );
    const __m128i color = _mm_or_si128( _mm_or_si128(rr,gg),bb );

    return( color );
}

