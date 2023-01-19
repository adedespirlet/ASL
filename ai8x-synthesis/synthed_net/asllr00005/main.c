/******************************************************************************
* Copyright (C) 2019-2022 Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// asllr00005
// Created using ai8xize.py --test-dir synthed_net --prefix asllr00005 --checkpoint-file trained/asl_0_0005lr-q.tar --config-file networks/aslnet.yaml --sample-input tests/sample_asl.npy --softmax --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --verbose --overwrite

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"
#include "camera.h"
#include "board.h"
#include "led.h"
#include "dma.h"


#define IMAGE_XRES  28											// X-resolution
#define IMAGE_YRES  28											// Y-resolution
#define CAMERA_FREQ 10000000	

volatile uint32_t cnn_time; // Stopwatch

// Data input: HWC 3x32x32 (3072 bytes total / 1024 bytes per channel):
static const uint32_t input_0[] = SAMPLE_INPUT_0;

int check_output(void)
{
  // int i;
  // uint32_t mask, len;
  // volatile uint32_t *addr;
  // const uint32_t *ptr = sample_output;

  // while ((addr = (volatile uint32_t *) *ptr++) != 0) {
  //   mask = *ptr++;
  //   len = *ptr++;
  //   for (i = 0; i < len; i++)
  //     if ((*addr++ & mask) != *ptr++) {
  //       printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
  //              i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
  //       return CNN_FAIL;
  //     }
  // }

  return CNN_OK;
}
void load_input(uint8_t* RawData, bool print_photo)
{
  int i;
  //const uint32_t *in0 = input_0;
  uint8_t* data_ptr = RawData;
  uint8_t ur, ug, ub, uzero;      // Colors
  uint32_t grey;
  uint8_t* data = NULL;
  //uzero = 254; ur = 31; ug = 83; ub = 63;
  //printf("0x%-8x,", 0xFFFFFFFF & ((uzero << 24) | (ur << 16) | (ug << 8) | ub)); // 0xfe1f533f
  //printf("0x%-8x,", 0xFFFFFFFF & ((ub << 24) | (ur << 16) | (ug << 8) | uzero)); // 0x3f1f53fe 

  for (i = 0; i < 784; i++)     // for each pixel // RGB 888
  {
    ur = data_ptr[ i*4+0 ]; 
    ug = data_ptr[ i*4+1 ];
    ub = data_ptr[ i*4+2 ];
    uzero = data_ptr[ i*4+3 ];
    //grey= 0x000000FF & (((ur) + (ug) + ub)/3);
    //grey= 0xFFFFFFFF & (((ur) + (ug) + ub)/3);
    grey= (((ur) + (ug) + ub)/3);
    if (print_photo)
    {
      printf("0x%.2x%2x%2x%2x, ",uzero,ur,ug,ub);
      if ((i+1) % 32 == 0) printf("\\\n");
    }
    
    //printf("0x%.2x%2x%2x%2x, ",uzero,ur,ug,ub);
    //printf("0x%-8x|", 0xFFFFFFFF & ((uzero << 24) | (ur << 16) | (ug << 8) | ub));
    //printf("0x000000%-8x,", grey);
    //printf("0x00%-8x,", 0x00FFFFFF & ((ur << 16) | (ug << 8) | ub));
    //printf("%x,",grey);
    //printf("0x%-8x|", 0xFFFFFFFF & ((uzero << 24) | (ur << 16) | (ug << 8) | ub));
    // Remove the following line if there is no risk that the source would overrun the FIFO:
    while ((data = get_camera_stream_buffer()) == NULL) {
            if (camera_is_image_rcv()) {
                break;
            }
    }
    memcpy32((uint32_t *) 0x50400000, grey, 784);
    //while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
    //*((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0
    //*((volatile uint32_t*) 0x50000008) = 0x00FFFFFF & ((ur << 16) | (ug << 8) | ub);
    //*((volatile uint32_t*) 0x50400000) = 0x000000FF & ((ur) | (ug) | ub);
    //printf("0x00%-8x,", 0x00FFFFFF & ((ur << 16) | (ug << 8) | ub));
  }
}

/*
void load_input_test( int i_)
{
  // This function loads the sample data input -- replace with actual data

  int i;
  const uint32_t *in0 = input_0;

  for (i = 0; i < 1024; i++) {
    // Remove the following line if there is no risk that the source would overrun the FIFO:
    while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); // Wait for FIFO 0
    *((volatile uint32_t *) 0x50000008) = *in0++; // Write FIFO 0
  }
}  
*/

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

int main(void)
{
  int i;
  int digs, tens;

  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("\nMAX78000 Starting up...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // camera
  int dma_channel;
  uint8_t*  RawData;
  uint32_t  ImageLength, ImageWidth, ImageHeight;
    
  MXC_DMA_Init();												// Initialize the DMA
  dma_channel = MXC_DMA_AcquireChannel();						// DMA channel

  camera_init(CAMERA_FREQ);							// Initialize the camera

  //
  // Setup the camera properties: camera resolution,pixel format,FIFO byte mode,DMA channel
  //
  camera_setup( IMAGE_XRES,         // width
                IMAGE_YRES,         // heigth
                PIXFORMAT_RGB888,   // pixel format
                FIFO_THREE_BYTE,     // FIFO mode
                USE_DMA,            // Set streaming mode
                dma_channel         // Allocate the DMA channel retrieved in initialization
                );
  
  
  printf("********** Camera is ready **********\r\n");
  while(1)
  {    // Main loop
    printf("********** Press PB1(SW1) to capture an image **********\r\n");
    bool print_photo = TRUE;
    while (!PB_Get(0)){
      if (PB_Get(1)){
        print_photo = 0;
        printf("Now it won't display the data on screen\n");
        MXC_Delay(MSEC(500));
      }
    }
  
    LED_Toggle(LED2);   
    camera_start_capture_image();								// Capture image
    //
    // Capture an image and display its properties
    //
    while(1)      // wait until picture is taken
    {
      if (camera_is_image_rcv()) 								// If image received...
      {
        camera_get_image(&RawData, &ImageLength, &ImageWidth, &ImageHeight);
        printf("Camera:\nraw=%x imglen=%d w=%d h=%d\n", RawData,ImageLength,ImageWidth,ImageHeight);
        MXC_Delay(SEC(1));
        break;
      }
    }
    LED_Toggle(LED2);
    printf("Picture taken\n");



    // Enable peripheral, enable CNN interrupt, turn on CNN clock
    // CNN clock: APB (50 MHz) div 1
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

    printf("\n*** CNN Inference with cifar10 ***\n");

    cnn_init(); // Bring state machine into consistent state
    cnn_load_weights(); // Load kernels
    cnn_load_bias();
    cnn_configure(); // Configure state machine
    cnn_start(); // Start CNN processing
    load_input(RawData, print_photo); // Load data input via FIFO

    while (cnn_time == 0)
    {
      MXC_LP_EnterSleepMode(); // Wait for CNN
    }

  if (check_output() != CNN_OK) fail();
  softmax_layer();

  printf("\n*** PASS ***\n\n");

#ifdef CNN_INFERENCE_TIMER
  printf("Approximate inference time: %u us\n\n", cnn_time);
#endif

  cnn_disable(); // Shut down CNN clock, disable peripheral

  printf("Classification results:\n");
  for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
    digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
    tens = digs % 10;
    digs = digs / 10;
    printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
  }

  return 0;
}
}


/*
  SUMMARY OF OPS
  Hardware: 737,924 ops (717,612 macc; 14,008 comp; 6,304 add; 0 mul; 0 bitwise)
    Layer 0: 54,000 ops (48,600 macc; 5,400 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 258,720 ops (254,016 macc; 4,704 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 131,424 ops (124,416 macc; 2,304 comp; 4,704 add; 0 mul; 0 bitwise)
    Layer 3: 232,000 ops (230,400 macc; 1,600 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 1,600 ops (0 macc; 0 comp; 1,600 add; 0 mul; 0 bitwise)
    Layer 5: 48,000 ops (48,000 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 10,080 ops (10,080 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 2,100 ops (2,100 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 63,726 bytes out of 442,368 bytes total (14%)
  Bias memory:   229 bytes out of 2,048 bytes total (11%)
*/

