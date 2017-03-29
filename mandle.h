#ifndef MANDLE_H
#define MANDLE_H


//const vars
const int XPICRES = 1000;
const int YPICRES = 1000;

void GenerateMandle(uint* buffer, int yStart, int yEnd);
void MandleSavePPM(uint* MandleBuffer);
uint* GenMandleSSEOMPCache(int yStart, int yEnd, int cachesize, int localbuffersize);
uint* GenerateMandleCache(int yStart, int yEnd, int cachesize, int localbuffersize);

#endif //MANDLE END
