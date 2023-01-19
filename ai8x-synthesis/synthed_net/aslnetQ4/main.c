/*******************************************************************************
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

// aslnetQ4
// Created using ai8xize.py --test-dir synthed_net --prefix aslnetQ4 --checkpoint-file trained/aslnetQ4-q.tar --config-file networks/aslnet.yaml --sample-input tests/sample_asl.npy --softmax --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --verbose --overwrite

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

#define IMAGE_SIZE_X (28)
#define IMAGE_SIZE_Y (28)
#define CAMERA_FREQ (5 * 1000 * 1000)

volatile uint32_t cnn_time; // Stopwatch

#ifdef USE_SAMPLEDATA
// Data input: HWC 3x128x128 (49152 bytes total / 16384 bytes per channel):
static const uint32_t input_0[] = SAMPLE_INPUT_0; // input data from header file
#else
static uint32_t input_0[IMAGE_SIZE_X * IMAGE_SIZE_Y]; // buffer for camera image
#endif

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// 1-channel 28x28 data input (784 bytes / 196 32-bit words):
// HWC 28x28, channels 0 to 0
//static const uint32_t input_0[] = SAMPLE_INPUT_0;

void load_input(void)
{
  for (int i= 0; i < 784; i++)
  {
    printf(" %i,\t", input_0[i]);
    /* code */
  }
  // This function loads the sample data input -- replace with actual data
  
  memcpy32((uint32_t *) 0x50400000, input_0, 784);
}

// Expected output of layer 7 for aslnetQ4 given the sample input (known-answer test)
// Delete this function for production code
//static const uint32_t sample_output[] = SAMPLE_OUTPUT;
int check_output(void)
{
  //int i;
  //uint32_t mask, len;
  //volatile uint32_t *addr;
  //const uint32_t *ptr = sample_output;

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

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_shift_q17p14_q15((q31_t *) ml_data, CNN_NUM_OUTPUTS, 4, ml_softmax);
}


void capture_process_camera(void)
{
    uint8_t* raw;
    uint32_t imgLen;
    uint32_t w, h;

    int cnt = 0;

    uint8_t r, g, b, uzero;
    uint16_t rgb;
    int j = 0;

    uint8_t* data = NULL;
    stream_stat_t* stat;
    uint32_t grey;

    camera_start_capture_image();

    // Get the details of the image from the camera driver.
    camera_get_image(&raw, &imgLen, &w, &h);
    printf("%i",imgLen);
    // Get image line by line
    for (int row = 0; row < h; row++) {
        // Wait until camera streaming buffer is full
        while ((data = get_camera_stream_buffer()) == NULL) {
            if (camera_is_image_rcv()) {
                break;
            }
        }

        //LED_Toggle(LED2);
        //printf("0x%.2x%2x%2x%2x, ",uzero,r,g,b);
        j = 0;
        
        for (int k = 0; k <  w; k ++) {
            // data format: 0x00bbggrr
            r = data[4*k];
            g = data[4*k + 1];
            b = data[4*k + 2];
            uzero= data[4*k+3];
            //skip k+3
            grey=  0x000000FF & ((r + g + b)/3);
            input_0[cnt++]=  grey;
            printf("0x%.2x%2x%2x%2x, \n",uzero,r,g,b);
            //0x00FFFFFF & ((ur << 16) | (ug << 8) | ub);
            // change the range from [0,255] to [-128,127] and store in buffer for CNN
            //input_0[cnt++] =  r ^ 0x00808080;
            //input_0[cnt++] =  r ^ 0x00000080;
            //input_0[cnt++] = ((b << 16) | (g << 8) | r) ^ 0x00808080;

        }
        
        //LED_Toggle(LED2);
        // Release stream buffer
        release_camera_stream_buffer();
    }

    //camera_sleep(1);
    stat = get_camera_stream_statistic();

    if (stat->overflow_count > 0) {
        printf("OVERFLOW DISP = %d\n", stat->overflow_count);
        LED_On(LED2); // Turn on red LED if overflow detected
        while (1)
            ;
    }
}



int main(void)
{

  FILE *fileName;
  fileName = fopen("/home/aurore/Documents/MLonMCUs/MAX78000/ai8x-synthesis/requirements.txt","r");

  if (fileName == NULL){
    printf("could not open file");
  }
  
  int dma_channel;
  int ret= 0;
  // Wait for PMIC 1.8V to become available, about 180ms after power up.
  MXC_Delay(200000);
  /* Enable camera power */
  Camera_Power(1);
    //MXC_Delay(300000);
  printf("\n\nCats-vs-Dogs Feather Demo\n");

  int i;
  int digs, tens;

  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // DO NOT DELETE THIS LINE:
  MXC_Delay(SEC(2)); // Let debugger interrupt if needed

  // Enable peripheral, enable CNN interrupt, turn on CNN clock
  // CNN clock: APB (50 MHz) div 1
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("\n*** CNN Inference Test ***\n");

  cnn_init(); // Bring state machine into consistent state
  cnn_load_weights(); // Load kernels
  cnn_load_bias();
  cnn_configure(); // Configure state machine



  // Initialize DMA for camera interface
  MXC_DMA_Init();
  dma_channel = MXC_DMA_AcquireChannel();


  // Initialize camera.
  printf("Init Camera.\n");
  camera_init(CAMERA_FREQ);

  printf("Configuring camera\n");
  ret = camera_setup(IMAGE_SIZE_X,              // width
               IMAGE_SIZE_Y,              // height
               PIXFORMAT_RGB888,   // pixel format
               FIFO_THREE_BYTE, // FIFO mode (four bytes is suitable for most cases)
               STREAMING_DMA,   // DMA (enabling DMA will drastically decrease capture time)
               dma_channel// Allocate the DMA channel retrieved in initialization
  );

  if (ret != STATUS_OK) {
        printf("Error returned from setting up camera. Error %d\n", ret);
        return -1;
  }
  camera_write_reg(0x11, 0x3); // set camera clock prescaller to prevent streaming overflow
  
  printf("********** Press PB1(SW1) to capture an image **********\r\n");
  while (!PB_Get(0))
        ;


  // Enable CNN clock
  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);
#ifdef USE_SAMPLEDATA
#ifdef TFT_ENABLE
      display_sampledata();
#endif
#else
      capture_process_camera();
#endif


  load_input(); // Load data input

  // for (int i= 0; i < 784; i++)
  // {
  //   fprintf(fileName, "%i\n", input_0[i]);
  //   /* code */
  // }
  //fclose(fileName);

  cnn_start(); // Start CNN processing

  SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0
  while (cnn_time == 0)
    __WFI(); // Wait for CNN

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
  Weight memory: 31,863 bytes out of 442,368 bytes total (7%)
  Bias memory:   273 bytes out of 2,048 bytes total (13%)
*/

