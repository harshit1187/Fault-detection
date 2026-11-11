/* fd_preprocess.h - firmware preprocessing for tinycnn_int8.onnx (spec v1)
 * Matches Harshit's norm.json and preprocess_ref.py.
 * Input : time-ordered raw windows (ring buffer already unrolled), 1024 samples each
 * Output: int8 [4][1024], channel-first (ax, ay, az, current), quantized with the model's x_scale / x_zp
 * Needs the FPU (F401RE: Cortex-M4F). Build with -ffp-contract=off if you want bit-exact match to Python.
 */
#ifndef FD_PREPROCESS_H
#define FD_PREPROCESS_H
#include <stdint.h>
#include <math.h>

#define FD_WINDOW        1024
#define FD_CHANNELS      4
#define FD_ACCEL_LSB_G   16384.0f                 /* MPU6050 +/-2 g */
#define FD_ADC_TO_AMPS   (3.3f / 4095.0f / 1.0f) /* 12-bit ADC, 3.3 V ref, 1 ohm shunt */
#define FD_X_SCALE       0.03954191f              /* model input quantization (x_scale) */
#define FD_X_ZP          0                        /* model input zero point */

static const float FD_NORM_MEAN[FD_CHANNELS] = { 1.202806743e-10f, -6.389200280e-11f, -1.486682777e-09f, 5.314997435e-01f };
static const float FD_NORM_STD [FD_CHANNELS] = { 4.577530921e-02f, 4.572412372e-02f, 4.580672458e-02f, 5.823453888e-02f };

static inline int8_t fd_quant(float x)
{
    long q = lrintf(x / FD_X_SCALE) + FD_X_ZP;   /* round-half-to-even, same as numpy/onnxruntime */
    if (q > 127)  q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static inline void fd_preprocess_int8(const int16_t *ax, const int16_t *ay, const int16_t *az,
                                      const uint16_t *cur, int8_t *out /* [4*FD_WINDOW] */)
{
    const int16_t *acc[3] = { ax, ay, az };
    for (int c = 0; c < 3; c++) {
        int32_t sum = 0;
        for (int i = 0; i < FD_WINDOW; i++) sum += acc[c][i];
        const float mean_raw = (float)sum / (float)FD_WINDOW;          /* per-window mean removal */
        for (int i = 0; i < FD_WINDOW; i++) {
            float g = ((float)acc[c][i] - mean_raw) / FD_ACCEL_LSB_G;
            out[c * FD_WINDOW + i] = fd_quant((g - FD_NORM_MEAN[c]) / FD_NORM_STD[c]);
        }
    }
    for (int i = 0; i < FD_WINDOW; i++) {                                 /* current: no mean removal */
        float amps = (float)cur[i] * FD_ADC_TO_AMPS;
        out[3 * FD_WINDOW + i] = fd_quant((amps - FD_NORM_MEAN[3]) / FD_NORM_STD[3]);
    }
}
#endif
