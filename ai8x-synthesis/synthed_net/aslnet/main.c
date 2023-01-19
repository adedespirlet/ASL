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

// aslnet
// Created using ai8xize.py --test-dir synthed_net --prefix aslnet --checkpoint-file trained/aslnet_trained2-q.pth.tar --config-file networks/aslnet.yaml --sample-input tests/sample_asl.npy --softmax --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --verbose --overwrite

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"


#include "board.h"
#include "led.h"
#include "dma.h"

volatile uint32_t cnn_time; // Stopwatch

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// 1-channel 28x28 data input (784 bytes / 196 32-bit words):
// HWC 28x28, channels 0 to 0
//static const uint32_t input_0[] = SAMPLE_INPUT_22;


// HWC 48x48, channels 4 to 7
//static const uint32_t input_22[] = SAMPLE_INPUT_22;
static const uint32_t input_0[] = SAMPLE_INPUT_12;
char sample[]="N";
// HWC 48x48, channels 8 to 11
//static const uint32_t input_23[] = SAMPLE_INPUT_23;

// HWC 48x48, channels 12 to 15
//static const uint32_t input_12[] = SAMPLE_INPUT_12;

// HWC 48x48, channels 16 to 19
//static const uint32_t input_1[] = SAMPLE_INPUT_1;


//static const uint32_t input_15[] = SAMPLE_INPUT_15;


void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, input_0, 784);
}

// Expected output of layer 7 for aslnet given the sample input (known-answer test)
// Delete this function for production code
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
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

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

// void select_sample(void) {
//   printf("Press Button PB1(SW1) to change sample and PB0 (SW0) to accept choice: ");
//   printf("current sample is: letter  a");
//   input_0 = SAMPLE_INPUT_1;
//   while (!PB_Get(0)){
//     //printf("sample chosen: Letter W");
//     //static const uint32_t input_0[] = SAMPLE_INPUT_22;
//       if (PB_Get(1)){
//         printf("sample chosen: Letter X");
//         input_0[784] = SAMPLE_INPUT_23;
//         MXC_Delay(MSEC(500));
//         while (!PB_Get(0)){
//           if (PB_Get(1)){
//           //printf("sample chosen: Letter N");
//           input_0[784] = SAMPLE_INPUT_12;
//           MXC_Delay(MSEC(500));
//           while (!PB_Get(0)){
//           }
//           break;
//         }
//         break;
//     }
//   }
    
// }
  

  //static const uint32_t input_0[] = SAMPLE_INPUT_23;

//}

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

int main(void)
{
  int i;
  int digs, tens;
  //select_sample();
  MXC_ICC_Enable(MXC_ICC0); // Enable cache

  // Switch to 100 MHz clock
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();


  printf("American Sign Language recognition\n ");

  printf("Waiting... Sample Data Chosen: Letter %s\n",sample);

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
  load_input(); // Load data input
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
  Weight memory: 63,726 bytes out of 442,368 bytes total (14%)
  Bias memory:   229 bytes out of 2,048 bytes total (11%)
*/

