///Mandlebrot OPENMPI
///Coded by MJ Wright 2014, 201176962
///Current implementation uses Master slave model
///Node 0 is master and allocates work to slaves

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include "mpi.h"
#include "mandle.h"

static void DebugWait2(int rank, int size) {
    char a;
    MPI_Bcast(&a, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
    printf("%d: DEBUG reached,%i\n", rank,size);
}



int main(int argc,char* argv[])
{

    //MPI
    int id;//process id
    int p;//num processes
    MPI_Status status;

    //Global vars
    int globalNumberofpixels = XPICRES*YPICRES;
   
    uint* mandleBuffer;
    
    //even vs odd?
    int lastProcessBlockSize;

    //local vars
    uint* localBuffer;
    int localblocksize;


    //Initialise OPENMPI
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    
    
    //globaltimer
    struct timeb gStart, gEnd;
    float tt;
    ftime(&gStart);//start global timer
    
	 //ssetup local timers
    struct timeb tstart, tend, cStart, cEnd;
	float diff;

	//split up problem size and calculate share of work
	int e = YPICRES/(p-1);
	localblocksize = e;
	
	//int f = d-(e*(p-1));
	//printf("f:%lli",f);
	
    //if(id!= (p-1)){//testing for uneven splits
    //    localblocksize = e;
    //}else{
    //    localblocksize = e;
    //}
 			          			
       //for calling the algorithm for each individual process
    int yStart = localblocksize*(id-1);//where x starts for each process, p=0 = 0 till p=4=last position
    int yEnd = yStart + localblocksize;//runs till here
    int arrSize = localblocksize*XPICRES;
	
	
	//start timers
	ftime(&tstart);

    MPI_Barrier(MPI_COMM_WORLD);//halt to ensure synchronizations
    
    if (id == 0){ //0 node is master and allocates mem 1nce
		
        mandleBuffer = (uint*)malloc(sizeof(uint)*globalNumberofpixels+1);//mandle buffer

        if(mandleBuffer == NULL){//cleanup on alloc fail
            printf("Failed to allocate memory for Mandlebuffer on process:%i\nExiting...\n",id);
            fflush(stdout);
            MPI_Finalize();
        }
        
        //now master process must recieve replies from nodes
        MPI_Status status;
		ftime(&cStart);//computation time!!
        int g;
        for(g = 1; g<=p-1; g++){//node 1->p-1
								
            int bufsize;
			
            MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);//probe for incoming sizes of buffers
			 // DebugWait(id);
            MPI_Get_count(&status, MPI_UNSIGNED, &bufsize);//get size
          
            uint* recvBuf = (uint*)malloc(sizeof(uint) * bufsize);//aloc mem to buffer to recieve
            
            MPI_Recv(recvBuf, bufsize, MPI_UNSIGNED, g, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);//recieve buffer
             
			//DebugWait(id, bufsize);
			//place buffer in global buffer
			int i;
            for(i=0; i<=bufsize; i++){
					mandleBuffer[i+(bufsize*(g-1))] = recvBuf[i];//place in array
            }
            
            free(recvBuf);
        }
		 ftime(&cEnd);//computation end timer!!!
		
    }//rest of nodes now do individual processing
    else{
		ftime(&cStart);//computation time!!
        localBuffer = GenMandleSSEOMPCache(yStart, yEnd, 256*4, arrSize); 
        MPI_Send(localBuffer, arrSize, MPI_UNSIGNED, 0,0, MPI_COMM_WORLD);//send answer
        //DebugWait(id, arrSize);
        ftime(&cEnd);//computation end timer!!!
    }
/*------------------------------------------------*///DebugWait(id);
	
	ftime(&tend);
    MPI_Barrier(MPI_COMM_WORLD);//wait until all is done
   
    //cleanup
    

    
    if(id==0){
		printf("\n\n%i)---Mandle Picture Computation done---\n",id);
		printf("%i)Escape time algorithm used with %i nodes used to compute\n",id,p-1);
		printf("%i)Image size: %i by %i\n",id,XPICRES,YPICRES);
		printf("%i)Runtime results below.\n\n",id);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);//standardize output
    //general time per process
	//diff =((float) (1000.0 * (cEnd.time - cStart.time) + (cEnd.millitm - cStart.millitm)))/1000.0;
	//printf("%i)Computation Time:%f\n",id,diff);
	//diff =((float) (1000.0 * (tend.time - tstart.time) + (tend.millitm - tstart.millitm)))/1000.0;
	//printf("%i)Total Time:%f\n\n",id,diff);
    
    
    
    if(id==0){//FPS approx
        ftime(&gEnd);
        tt=((float) (1000.0 * (cEnd.time - cStart.time) + (cEnd.millitm - cStart.millitm)))/1000.0;
		printf("\n\n[------------------------Total Running time:%f---------------------------]\n",tt);
		tt = 1/tt;//rough FPS estimate as 1/tt = frequency of its occurance
		printf("[------------------------Rough FPS estimate:%f---------------------------]\n\n",tt);
	}
    
    
    
    //cleanup & Save!
    if(id==0){
        MandleSavePPM(mandleBuffer);
        printf("%i)Image saved to result.ppm\n\n",id);
        free(mandleBuffer);
    }
	if(id!=0)
		free(localBuffer);
		
	
    
    MPI_Finalize();

    return 0;
}

