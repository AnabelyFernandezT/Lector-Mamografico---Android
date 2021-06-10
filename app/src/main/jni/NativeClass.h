
#include <jni.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>



using namespace cv;
using namespace std;

#ifndef TRABIMG_NATIVECLASS_H
#define TRABIMG_NATIVECLASS_H

extern "C" {
JNIEXPORT jintArray JNICALL
Java_com_example_trabimg_MainActivity_grayProc(JNIEnv *env, jobject instance,
                                               jintArray buf, jint w, jint h);

JNIEXPORT void JNICALL
Java_com_example_trabimg_MainActivity_watershed(JNIEnv *env, jobject instance, jstring filename_,
                                                jlong salida);

float var(int hist[],int level,float val,int pix_num );
void metodoOtsu1Umbral(Mat imagenORiginal, int canal, Mat& out);
/// Calcula el histograma normalizado
void histogramaNormalizado( float p_i[ ], float n_i[ ], float N ,int nivelDeIntensidad );
/// Calcula las sumas comulativas P1(k)
float obtenerP1k( int k, float p_i[ ] );
/// Calcula las medias comulativas m(k)
float obtenermk( int k, float p_i[ ] );
/// Calcula la intensidad media global mg
float obtenermg( int nivelDeIntensidad, float p_i[ ] );
/// Calcula la varianza entre clases
float varianzaEntreClases( float P1k, float mk, float mg );
/// Calcula el umbral de Otsu k*
int obtenerUmbralOptimo(float p_i[ ], float varianza, float mg , int kInicial ,int nivelDeIntensidad);
/// Binariza la matriz BGR
void binarizarMatriz( Mat imagen1, int kOtsu, Mat& out );

Mat watershedSegment(Mat & image, int & noOfSegments);
Mat createSegmentationDisplay(Mat & segments,int numOfSegments,Mat & image);
void mergeSegments(Mat & image,Mat & segments, int & numOfSegments);
}

#endif //TRABIMG_NATIVECLASS_H
