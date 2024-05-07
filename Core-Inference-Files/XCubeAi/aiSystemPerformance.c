/**
 ******************************************************************************
 * @file    aiSystemPerformance.c
 * @author  MCD/AIS Team
 * @brief   Entry points for AI system performance application (multiple network)
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2019,2021 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software is licensed under terms that can be found in the LICENSE file in
 * the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

/*
 * Description:
 *
 * - Simple STM32 application to measure and report the system performance of
 *   a generated NN
 * - Use the multiple-network API
 * - Random input values are injected in the NN to measure the inference time
 *   and to monitor the usage of the stack and/or the heap. Output value are
 *   skipped.
 * - After N iterations (_APP_ITER_ C-define), results are reported through a
 *   re-target printf
 * - aiSystemPerformanceInit()/aiSystemPerformanceProcess() functions should
 *   be called from the main application code.
 * - Only UART (to re-target the printf) & CORE clock setting are expected
 *   by the initial run-time (main function).
 *   CRC IP should be also enabled
 *
 * STM32CubeIDE (GCC-base toolchain)
 *  - Linker options "-Wl,--wrap=malloc -Wl,--wrap=free" should be used
 *    to support the HEAP monitoring
 *
 * TODO:
 *  - (nice-to_have) add HEAP monitoring for IAR tool-chain
 *  - (nice-to-have) add HEAP/STACK monitoring MDK-ARM Keil tool-chain
 *
 * History:
 *  - v1.0 - Initial version
 *  - v1.1 - Complete minimal interactive console
 *  - v1.2 - Adding STM32H7 MCU support
 *  - v1.3 - Adding STM32F3 MCU support
 *  - v1.4 - Adding Profiling mode
 *  - v2.0 - Adding Multiple Network support
 *  - v2.1 - Adding F3 str description
 *  - v3.0 - Adding FXP support
 *           Adding initial multiple IO support (legacy mode)
 *           Removing compile-time STM32 family checking
 *  - v3.1 - Fix cycle count overflow
 *           Add support for external memory for data activations
 *  - v4.0 - Adding multiple IO support
 *  - v4.1 - Adding L5 support
 *  - v4.2 - Adding support for inputs in activations buffer
 *  - v4.3 - Fix - fill input samples loop + HAL_delay report
 *  - v4.4 - Complete dev_id str description
 *  - v5.0 - Add inference time by layer (with runtime observer API support)
 *           Improve reported network info (minor)
 *           Fix stack calculation (minor)
 *  - v5.1 - Create separate C-files (aiTestUtility/aiTestHelper) for common (non AI)
 *           functions with aiValidation firmware.
 *           Align aiBootstrap/aiInit/aiDeInit functions with aiValidation.c
 *           Adding support for outputs in activations buffer.
 *  - v5.2 - Use the fix cycle count overflow support for time per layer
 *  - v5.3 - Add support to use SYSTICK only (remove direct call to DWT fcts)
 *  - v6.0 - Update with new API to support the fragmented activations/weights buffer
 *           activations and io buffers are fully handled by app_x-cube-ai.c/h files
 *           Align code with the new ai_buffer struct definition
 *  - v7.0 - align code with ai_stm32_adpaptor.h file (remove direct call of HAL_xx fcts)
*/

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "yoloOutput.h"

#if !defined(USE_OBSERVER)
#define USE_OBSERVER         1 /* 0: remove the registration of the user CB to evaluate the inference time by layer */
#endif

#define USE_CORE_CLOCK_ONLY  0 /* 1: remove usage of the HAL_GetTick() to evaluate the number of CPU clock. Only the Core
                                *    DWT IP is used. HAL_Tick() is requested to avoid an overflow with the DWT clock counter
                                *    (32b register) - USE_SYSTICK_ONLY should be set to 0.
                                */
#define USE_SYSTICK_ONLY     0 /* 1: use only the SysTick to evaluate the time-stamps (for Cortex-m0 based device, this define is forced) */


#if !defined(APP_DEBUG)
#define APP_DEBUG     	     0 /* 1: add debug trace - application level */
#endif

/* APP header files */
#include <aiSystemPerformance.h>
#include <aiTestUtility.h>
#include <aiTestHelper.h>
#include "output_array_3.h"

/* AI Run-time header files */
#include "ai_platform_interface.h"


/* -----------------------------------------------------------------------------
 * TEST-related definitions
 * -----------------------------------------------------------------------------
 */


#define _APP_VERSION_MAJOR_     (0x07)
#define _APP_VERSION_MINOR_     (0x00)
#define _APP_VERSION_   ((_APP_VERSION_MAJOR_ << 8) | _APP_VERSION_MINOR_)

#define _APP_NAME_      "AI system performance measurement"
#define _APP_ITER_       16  /* number of iteration for perf. test */


/* Global variables */
static bool observer_mode = true;
static bool profiling_mode = false;
static int  profiling_factor = 5;


/*-------- For YOLOv3 anchors[3][3][2]-------------*/
/*
 * const int anchors[2][3][2] = {
 * {{10,13},{16,30},{33,23},},
{{30,61},{62,45},{59,119},},
{{116,90},{156,198},{373,326},},
};
 */

/*-------- Set the number and size of anchors-------------*/
const int anchors[2][3][2] = {
{{23,27},{37,58},{81,82},},
{{81,82},{135,169},{344,319},},
};
void det_post_processing(ai_float* first, ai_float* second, int* box_cnt, det_box** ret_box, float threshold);

/* -----------------------------------------------------------------------------
 * AI-related functions
 * -----------------------------------------------------------------------------
 */

struct ai_network_exec_ctx {
  ai_handle handle;
  ai_network_report report;
} net_exec_ctx[AI_MNETWORK_NUMBER] = {0};

void det_post_processing(ai_float* first, ai_float* second, int* box_cnt, det_box** ret_box, float threshold){

	/*-------- Combine ouput arrays, for yolov3 its a third output ---------*/
	float* input_data[2] = {first, second,};
	/*-------- invoke postproccessing, all inputs must be adapted accordingly ---------*/
    postprocessing(input_data,128, 160, 18, 1, anchors, 2, 0.40f, threshold, box_cnt, ret_box);
}

static int aiBootstrap(struct ai_network_exec_ctx *ctx, const char *nn_name)
{
  ai_error err;

  /* Creating the instance of the  network ------------------------- */
  LC_PRINT("Creating the network \"%s\"..\r\n", nn_name);

  err = ai_mnetwork_create(nn_name, &ctx->handle, NULL);
  if (err.type) {
    aiLogErr(err, "ai_mnetwork_create");
    return -1;
  }

  /* Initialize the instance --------------------------------------- */
  LC_PRINT("Initializing the network\r\n");

  if (!ai_mnetwork_get_report(ctx->handle, &ctx->report)) {
    err = ai_mnetwork_get_error(ctx->handle);
    aiLogErr(err, "ai_mnetwork_get_info");
    ai_mnetwork_destroy(ctx->handle);
    ctx->handle = AI_HANDLE_NULL;
    return -2;
  }

  if (!ai_mnetwork_init(ctx->handle)) {
    err = ai_mnetwork_get_error(ctx->handle);
    aiLogErr(err, "ai_mnetwork_init");
    ai_mnetwork_destroy(ctx->handle);
    ctx->handle = AI_HANDLE_NULL;
    return -4;
  }

  /* Display the network info -------------------------------------- */
  if (ai_mnetwork_get_report(ctx->handle, &ctx->report)) {
    aiPrintNetworkInfo(&ctx->report);
  } else {
    err = ai_mnetwork_get_error(ctx->handle);
    aiLogErr(err, "ai_mnetwork_get_info");
    ai_mnetwork_destroy(ctx->handle);
    ctx->handle = AI_HANDLE_NULL;
    return -2;
  }

  return 0;
}

static int aiInit(void)
{
  int res = -1;
  const char *nn_name;
  int idx;

  aiPlatformVersion();

  /* Reset the contexts -------------------------------------------- */
  for (idx=0; idx < AI_MNETWORK_NUMBER; idx++) {
    net_exec_ctx[idx].handle = AI_HANDLE_NULL;
  }

  /* Discover and initialize the network(s) ------------------------ */
  LC_PRINT("Discovering the network(s)...\r\n");

  idx = 0;
  do {
    nn_name = ai_mnetwork_find(NULL, idx);
    if (nn_name) {
      LC_PRINT("\r\nFound network \"%s\"\r\n", nn_name);
      res = aiBootstrap(&net_exec_ctx[idx], nn_name);
      if (res)
        nn_name = NULL;
    }
    idx++;
  } while (nn_name);

  return res;
}

static void aiDeInit(void)
{
  ai_error err;
  int idx;

  /* Releasing the instance(s) ------------------------------------- */
  LC_PRINT("Releasing the instance(s)...\r\n");

  for (idx=0; idx<AI_MNETWORK_NUMBER; idx++) {
    if (net_exec_ctx[idx].handle != AI_HANDLE_NULL) {
      if (ai_mnetwork_destroy(net_exec_ctx[idx].handle)
          != AI_HANDLE_NULL) {
        err = ai_mnetwork_get_error(net_exec_ctx[idx].handle);
        aiLogErr(err, "ai_mnetwork_destroy");
      }
      net_exec_ctx[idx].handle = AI_HANDLE_NULL;
    }
  }
}


#if defined(USE_OBSERVER) && USE_OBSERVER == 1

struct u_node_stat {
  uint64_t dur;
  uint32_t n_runs;
};

struct u_observer_ctx {
  uint64_t n_cb;
  uint64_t start_t;
  uint64_t u_dur_t;
  uint64_t k_dur_t;
  struct u_node_stat *nodes;
};

static struct u_observer_ctx u_observer_ctx;

/* User callback */
static ai_u32 user_observer_cb(const ai_handle cookie,
    const ai_u32 flags,
    const ai_observer_node *node) {

  struct u_observer_ctx *u_obs;

  volatile uint64_t ts = cyclesCounterEnd(); /* time stamp entry */

  u_obs = (struct u_observer_ctx *)cookie;
  u_obs->n_cb += 1;

  if (flags & AI_OBSERVER_POST_EVT) {
    const uint64_t end_t = ts - u_obs->start_t;
    u_obs->k_dur_t += end_t;
    u_obs->nodes[node->c_idx].dur += end_t;
    u_obs->nodes[node->c_idx].n_runs += 1;
  }

  u_obs->start_t = cyclesCounterEnd();    /* time stamp exit */
  u_obs->u_dur_t += u_obs->start_t  - ts; /* accumulate cycles used by the CB */
  return 0;
}


void aiObserverInit(struct ai_network_exec_ctx *net_ctx)
{
  ai_handle  net_hdl;
  ai_network_params net_params;
  ai_bool res;
  int sz;

  if (!net_ctx || (net_ctx->handle == AI_HANDLE_NULL) || !net_ctx->report.n_nodes)
    return;


  /* retrieve real handle */
  ai_mnetwork_get_private_handle(net_ctx->handle, &net_hdl, &net_params);

  memset((void *)&u_observer_ctx, 0, sizeof(struct u_observer_ctx));

  /* allocate resources to store the state of the nodes */
  sz = net_ctx->report.n_nodes * sizeof(struct u_node_stat);
  u_observer_ctx.nodes = (struct u_node_stat*)malloc(sz);
  if (!u_observer_ctx.nodes) {
    LC_PRINT("W: enable to allocate the u_node_stats (sz=%d) ..\r\n", sz);
    return;
  }

  memset(u_observer_ctx.nodes, 0, sz);

  /* register the callback */
  res = ai_platform_observer_register(net_hdl, user_observer_cb,
      (ai_handle)&u_observer_ctx, AI_OBSERVER_PRE_EVT | AI_OBSERVER_POST_EVT);
  if (!res) {
    LC_PRINT("W: unable to register the user CB\r\n");
    free(u_observer_ctx.nodes);
    u_observer_ctx.nodes = NULL;
    return;
  }
}

extern const char* ai_layer_type_name(const int type);

void aiObserverDone(struct ai_network_exec_ctx *net_ctx)
{
  ai_handle  net_hdl;
  ai_network_params net_params;
  struct dwtTime t;
  uint64_t cumul;
  ai_observer_node node_info;

  if (!net_ctx || (net_ctx->handle == AI_HANDLE_NULL) ||
      !net_ctx->report.n_nodes || !u_observer_ctx.nodes)
    return;

  /* retrieve real handle */
  ai_mnetwork_get_private_handle(net_ctx->handle, &net_hdl, &net_params);

  ai_platform_observer_unregister(net_hdl, user_observer_cb,
      (ai_handle)&u_observer_ctx);

  LC_PRINT("\r\n Inference time by c-node\r\n");
  dwtCyclesToTime(u_observer_ctx.k_dur_t / u_observer_ctx.nodes[0].n_runs, &t);
  LC_PRINT("  kernel  : %d.%03dms (time passed in the c-kernel fcts)\r\n", t.s * 1000 + t.ms, t.us);
  dwtCyclesToTime(u_observer_ctx.u_dur_t / u_observer_ctx.nodes[0].n_runs, &t);
  LC_PRINT("  user    : %d.%03dms (time passed in the user cb)\r\n", t.s * 1000 + t.ms, t.us);
#if APP_DEBUG == 1
  LC_PRINT("  cb #    : %d\n", (int)u_observer_ctx.n_cb);
#endif

  LC_PRINT("\r\n %-6s%-20s%-7s  %s\r\n", "c_id", "type", "id", "time (ms)");
  LC_PRINT(" ---------------------------------------------------\r\n");

  cumul = 0;
  node_info.c_idx = 0;
  while (ai_platform_observer_node_info(net_hdl, &node_info)) {
    struct u_node_stat *sn = &u_observer_ctx.nodes[node_info.c_idx];
    const char *fmt;
    cumul +=  sn->dur;
    dwtCyclesToTime(sn->dur / (uint64_t)sn->n_runs, &t);
#if defined(AI_APP_LITE) && AI_APP_LITE == 1
    if ((node_info.type & (ai_u16)0x8000) >> 15)
      fmt = " %-6dTD-%-17d%-5d %6d.%03d %6.02f %c\r\n";
    else
      fmt = " %-6d%-20d%-5d %6d.%03d %6.02f %c\r\n";
#else
    if ((node_info.type & (ai_u16)0x8000) >> 15)
      fmt = " %-6dTD-%-17s%-5d %6d.%03d %6.02f %c\r\n";
    else
      fmt = " %-6d%-20s%-5d %6d.%03d %6.02f %c\r\n";
#endif

    LC_PRINT(fmt, node_info.c_idx,
#if defined(AI_APP_LITE) && AI_APP_LITE == 1
        node_info.type  & (ai_u16)0x7FFF,
#else
        ai_layer_type_name(node_info.type  & (ai_u16)0x7FFF),
#endif
        (int)node_info.id,
        t.s * 1000 + t.ms, t.us,
        ((double)u_observer_ctx.nodes[node_info.c_idx].dur * 100.0) / (double)u_observer_ctx.k_dur_t,
        '%');
    node_info.c_idx++;
  }

  LC_PRINT(" -------------------------------------------------\r\n");
  cumul /= u_observer_ctx.nodes[0].n_runs;
  dwtCyclesToTime(cumul, &t);
  LC_PRINT(" %31s %6d.%03d ms\r\n", "", t.s * 1000 + t.ms, t.us);

  free(u_observer_ctx.nodes);
  memset((void *)&u_observer_ctx, 0, sizeof(struct u_observer_ctx));

  return;
}
#endif


/* -----------------------------------------------------------------------------
 * Specific APP/test functions
 * -----------------------------------------------------------------------------
 */

static int aiTestPerformance(int idx)
{
  int iter;
  ai_i32 batch;
  int niter;

  struct dwtTime t;
  uint64_t tcumul;
  uint64_t tend;
  ai_macc cmacc;


  ai_buffer ai_input[AI_MNETWORK_IN_NUM];
  ai_buffer ai_output[AI_MNETWORK_OUT_NUM];

  if (net_exec_ctx[idx].handle == AI_HANDLE_NULL) {
    LC_PRINT("E: network handle is NULL\r\n");
    return -1;
  }

  MON_STACK_INIT();

  if (profiling_mode)
    niter = _APP_ITER_ * profiling_factor;
  else
    niter = _APP_ITER_;

  MON_STACK_CHECK0();

  /* reset/init cpu clock counters */
  tcumul = 0ULL;

  MON_STACK_MARK();

  if ((net_exec_ctx[idx].report.n_inputs > AI_MNETWORK_IN_NUM) ||
      (net_exec_ctx[idx].report.n_outputs > AI_MNETWORK_OUT_NUM))
  {
    LC_PRINT("E: AI_MNETWORK_IN/OUT_NUM definition are incoherent\r\n");
    port_hal_delay(100);
    return -1;
  }

  /*--------------- Fill the input tensor descriptors ---------------------------*/
  for (int i = 0; i < net_exec_ctx[idx].report.n_inputs; i++) {
    ai_input[i] = net_exec_ctx[idx].report.inputs[i];
    if (net_exec_ctx[idx].report.inputs[i].data)
      ai_input[i].data = AI_HANDLE_PTR(net_exec_ctx[idx].report.inputs[i].data);
    else
      ai_input[i].data = AI_HANDLE_PTR(data_ins[i]);
  }

  /*--------------- Fill/set the output tensor descriptors ---------------*/
  for (int i = 0; i < net_exec_ctx[idx].report.n_outputs; i++) {
    ai_output[i] = net_exec_ctx[idx].report.outputs[i];
    if (net_exec_ctx[idx].report.outputs[i].data)
      ai_output[i].data = AI_HANDLE_PTR(net_exec_ctx[idx].report.outputs[i].data);
    else
      ai_output[i].data = AI_HANDLE_PTR(data_outs[i]);
  }

  MON_ALLOC_RESET();

  /*------------------------- Main inference loop --------------------------------*/
  for (iter = 0; iter < niter; iter++) {

    /*----------------- Fill input tensors with image data -----------------------*/
    for (int i = 0; i < net_exec_ctx[idx].report.n_inputs; i++) {
      const ai_buffer_format fmt = AI_BUFFER_FORMAT(&ai_input[i]);
      ai_i8 *in_data = (ai_i8 *)ai_input[i].data;

      for (ai_size j = 0; j < AI_BUFFER_SIZE(&ai_input[i]); ++j) {
        if  (AI_BUFFER_FMT_GET_TYPE(fmt) == AI_BUFFER_FMT_TYPE_FLOAT) {
          *(ai_float *)(in_data + j * 4) = test_image_r[j];
        }
      }
    }


    MON_ALLOC_ENABLE();

    cyclesCounterStart();

    /*---------------------- Runs actual Inference -----------------------------*/
    batch = ai_mnetwork_run(net_exec_ctx[idx].handle, ai_input, ai_output);
    if (batch != 1) {
      aiLogErr(ai_mnetwork_get_error(net_exec_ctx[idx].handle),
          "ai_mnetwork_run");
      break;
    }

    LC_PRINT("Inference output..\n");


    /*----------------- For two output layers ---------------------------*/
    float *out_data_0 = (float *)ai_output[0].data;
    float *out_data_1 = (float *)ai_output[1].data;

    int box_count;
    det_box* detected_boxes;
    float threshold = 0.6f; // Set this to your desired threshold

    /*------------- Postprocessing modified from TinyEngine ----------------------*/
    det_post_processing(out_data_0, out_data_1, &box_count, &detected_boxes, threshold);

    /*----------------- Output of the bounding boxes ---------------------------*/
    for (int i = 0; i < box_count; i++) {
  	  LC_PRINT("Box %d: x0=%.2f, y0=%.2f, x1=%.2f, y1=%.2f, score=%.2f\r\n",
  			  i,
  			  detected_boxes[i].x0,
  			  detected_boxes[i].y0,
  			  detected_boxes[i].x1,
  			  detected_boxes[i].y1,
  			  detected_boxes[i].score);

    }

    tend = cyclesCounterEnd();

    MON_ALLOC_DISABLE();

    tcumul += tend;

    dwtCyclesToTime(tend, &t);
  } /* end of the main loop */



  MON_STACK_EVALUATE();

  LC_PRINT("\r\n");

  tcumul /= (uint64_t)iter;
  dwtCyclesToTime(tcumul, &t);
  MON_STACK_REPORT();
  MON_ALLOC_REPORT();
  return 0;
}

#define CONS_EVT_TIMEOUT    (0)
#define CONS_EVT_QUIT       (1)
#define CONS_EVT_RESTART    (2)
#define CONS_EVT_HELP       (3)
#define CONS_EVT_PAUSE      (4)
#define CONS_EVT_PROF       (5)
#define CONS_EVT_HIDE       (6)

#define CONS_EVT_UNDEFINED  (100)

static int aiTestConsole(void)
{
  uint8_t c = 0;

  if (ioRawGetUint8(&c, 5000) == -1) /* Timeout */
    return CONS_EVT_TIMEOUT;

  if ((c == 'q') || (c == 'Q'))
    return CONS_EVT_QUIT;

  if ((c == 'd') || (c == 'D'))
    return CONS_EVT_HIDE;

  if ((c == 'r') || (c == 'R'))
    return CONS_EVT_RESTART;

  if ((c == 'h') || (c == 'H') || (c == '?'))
    return CONS_EVT_HELP;

  if ((c == 'p') || (c == 'P'))
    return CONS_EVT_PAUSE;

  if ((c == 'x') || (c == 'X'))
    return CONS_EVT_PROF;

  return CONS_EVT_UNDEFINED;
}


/* -----------------------------------------------------------------------------
 * Exported/Public functions
 * -----------------------------------------------------------------------------
 */

int aiSystemPerformanceInit(void)
{
  LC_PRINT("\r\n#\r\n");
  LC_PRINT("# %s %d.%d\r\n", _APP_NAME_ , _APP_VERSION_MAJOR_,
      _APP_VERSION_MINOR_ );
  LC_PRINT("#\r\n");

  systemSettingLog();

  crcIpInit();
  cyclesCounterInit();

  aiInit();

  srand(3); /* deterministic outcome */

  return 0;
}

int aiSystemPerformanceProcess(void)
{
  int r;
  int idx = 0;
  r = aiTestPerformance(idx);

  return r;
}

void aiSystemPerformanceDeInit(void)
{
  LC_PRINT("\r\n");
  aiDeInit();
  LC_PRINT("bye bye ...\r\n");
}

