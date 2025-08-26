#define _GNU_SOURCE
#include <math.h>

double __log_finite(double x) { return log(x); }
double __exp_finite(double x) { return exp(x); }
double __pow_finite(double x, double y) { return pow(x, y); }
double __log10_finite(double x) { return log10(x); }
double __acos_finite(double x) { return acos(x); }
double __asin_finite(double x) { return asin(x); }
double __atan2_finite(double x, double y) { return atan2(x, y); }
double __cosh_finite(double x) { return cosh(x); }
double __sinh_finite(double x) { return sinh(x); }
float __powf_finite(float x, float y) { return powf(x, y); }

// Add other missing functions as needed.
double _ZGVbN2v___log_finite(double x) { return log(x); }
double _ZGVbN2v___exp_finite(double x) { return exp(x); }