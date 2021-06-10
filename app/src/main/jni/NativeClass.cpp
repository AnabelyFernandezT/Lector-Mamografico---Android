

#include "NativeClass.h"
#include <android/log.h>

#define  LOG_TAG    "NativeClass"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_trabimg_MainActivity_grayProc(JNIEnv *env, jobject instance,
                                                  jintArray buf, jint w, jint h) {

    jint *cbuf;
    jboolean ptfalse = false;
    cbuf = env->GetIntArrayElements(buf, &ptfalse);
    if(cbuf == NULL){
        return 0;
    }

    Mat imgData(h, w, CV_8UC4, (unsigned char*)cbuf);

    uchar* ptr = imgData.ptr(0);
    for(int i = 0; i < w*h; i ++){
        uchar grayScale = (uchar)(ptr[4*i+2]*0.299 + ptr[4*i+1]*0.587 + ptr[4*i+0]*0.114);
        ptr[4*i+1] = grayScale;
        ptr[4*i+2] = grayScale;
        ptr[4*i+0] = grayScale;
    }

    int size=w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, cbuf);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    return result;

}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_trabimg_MainActivity_watershed(JNIEnv *env, jobject instance, jstring filename_, jlong salida) {
    const char *filename = env->GetStringUTFChars(filename_, 0);

    Mat image; // imagen grayscale

    //Mat& sal = *(Mat*)salida;

    String path = filename;
    image = imread(path);

    int numOfSegments = 0;
    //Apply watershed
    Mat segments= watershedSegment(image,numOfSegments);
    //Merge segments in order to reduce over segmentation
    mergeSegments(image,segments, numOfSegments);
    //To display the merged segments
    Mat wshed = createSegmentationDisplay(segments,numOfSegments, image);
    //To display the merged segments blended with the image
    Mat& wshedWithImage =*(Mat*)salida;
        wshedWithImage = createSegmentationDisplay(segments,numOfSegments,image);



    env->ReleaseStringUTFChars(filename_, filename);


}

void mergeSegments(Mat & image,Mat & segments, int & numOfSegments)
{
    //To collect pixels from each segment of the image
    vector<Mat> samples;
    //In case of multiple merging iterations, the numOfSegments should be updated
    int newNumOfSegments = numOfSegments;

    //Initialize the segment samples
    for(int i=0;i<=numOfSegments;i++)
    {
        Mat sampleImage;
        samples.push_back(sampleImage);
    }

    //collect pixels from each segments
    for(int i = 0; i < segments.rows; i++ )
    {
        for(int j = 0; j < segments.cols; j++ )
        {
            //check what segment the image pixel belongs to
            int index = segments.at<int>(i,j);
            if(index >= 0 && index<numOfSegments)
            {
                samples[index].push_back(image(Rect(j,i,1,1)));
            }
        }
    }

    //create histograms
    vector<MatND> hist_bases;
    Mat hsv_base;
    /// Using 35 bins for hue component
    int h_bins = 35;
    /// Using 30 bins for saturation component
    int s_bins = 30;
    int histSize[] = { h_bins,s_bins };

    // hue varies from 0 to 256, saturation from 0 to 180
    float h_ranges[] = { 0, 256 };
    float s_ranges[] = { 0, 180 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the 0-th and 1-st channels
    int channels[] = { 0,1 };

    // To store the histograms
    MatND hist_base;
    for(int c=1;c<numOfSegments;c++)
    {
        if(samples[c].dims>0){
            //convert the sample to HSV
            cvtColor( samples[c], hsv_base, CV_BGR2HSV );
            //calculate the histogram
            calcHist( &hsv_base, 1, channels, Mat(), hist_base,2, histSize, ranges, true, false );
            //normalize the histogram
            normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
            //append to the collection
            hist_bases.push_back(hist_base);
        }else
        {
            hist_bases.push_back(MatND());
        }

        hist_base.release();
    }

    //To store the similarity of histograms
    double similarity = 0;
    //to keep the track of already merged segments
    vector<bool> mearged;
    //initialize the merged segments tracker
    for(int k = 0; k < hist_bases.size(); k++)
    {
        mearged.push_back(false);
    }

    //calculate the similarity of the histograms of each pair of segments
    for(int c=0;c<hist_bases.size();c++)
    {
        for(int q=c+1;q<hist_bases.size();q++)
        {
            //if the segment is not merged alreay
            if(!mearged[q])
            {
                if(hist_bases[c].dims>0 && hist_bases[q].dims>0)
                {
                    //calculate the histogram similarity
                    similarity = compareHist(hist_bases[c],hist_bases[q],CV_COMP_BHATTACHARYYA);
                    //if similay
                    if(similarity>0.8)
                    {
                        mearged[q]=true;
                        if(q!=c)
                        {
                            //reduce number of segments
                            newNumOfSegments--;
                            for(int i = 0; i < segments.rows; i++ )
                            {
                                for(int j = 0; j < segments.cols; j++ )
                                {
                                    int index = segments.at<int>(i,j);
                                    //merge the segment q with c
                                    if(index==q){
                                        segments.at<int>(i,j) = c;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    numOfSegments = newNumOfSegments;
}

void metodoOtsu1Umbral(Mat imagen1, int canal,Mat& out){

    int N; ///Numero de pixeles
    int nivelDeIntensidad = 256;
    float n_i[ nivelDeIntensidad ]; ///numero de pixeles con intensidad i
    int i, x, y;
    int kInicial = 100, kOtsu;
    float p_i[ nivelDeIntensidad ];
    float P1k = 0, mk = 0, mg = 0;
    float varianza;

    N = imagen1.rows * imagen1.cols;

    for ( i = 0; i < nivelDeIntensidad; i++ ) {
        n_i[ i ]=0;
    }

    /* if ( canal > 2 || canal < 0) {
         printf("\nEl canal de color esta fuera del rango [0-2]\nPista:\n0 -> Azul\n1 -> Verde\n2 -> Rojo\n");
         exit(EXIT_FAILURE);
     }*/

    for( int x = 0; x < imagen1.rows; x++ ){
        for( int y = 0; y < imagen1.cols; y++ ){

            n_i[ imagen1.at<uchar>(x,y) ] = n_i[ imagen1.at<uchar>(x,y) ] + 1; //para nivel de gris
            //n_i[ imagen1.at<Vec3b>(x, y)[canal] ] = n_i[ imagen1.at<Vec3b>( x, y )[ canal ] ] + 1;
        }
    }

    /// 1 calcula el histograma normalizado
    histogramaNormalizado( p_i, n_i, N, nivelDeIntensidad );
    /// 2 calcula las sumas comulativas P1(k)
    P1k = obtenerP1k( kInicial, p_i );
    /// 3 calcula las medias comulativas m(k)
    mk = obtenermk( kInicial, p_i );
    /// 4 calcula la intensidad media global
    mg = obtenermg( nivelDeIntensidad, p_i );
    /// 5 calcula la varianza entre clases
    varianza = varianzaEntreClases( P1k, mk, mg );
    /// 6 obtiene el umbral de otsu k*
    kOtsu = obtenerUmbralOptimo( p_i, varianza, mg , kInicial , nivelDeIntensidad );

    binarizarMatriz( imagen1, kOtsu,out );
}

void histogramaNormalizado( float p_i[], float n_i[], float N , int nivelDeIntensidad ){
    int i;
    for( i = 0; i < nivelDeIntensidad; i++){
        p_i[i] = n_i[i] / N;
    }
}

float obtenerP1k( int k, float p_i[ ] ){
    int i;
    float P1k = 0;

    for( i = 0; i <= k; i++){
        P1k = P1k + p_i[i];
    }

    return P1k;
}

float obtenermk( int k, float p_i[ ] ){
    int i;
    float mk = 0;

    for( i = 0; i <= k; i++){
        mk = mk + ( i * p_i[i] );
    }

    return mk;
}

float obtenermg( int nivelDeIntensidad, float p_i[ ] ){
    int i;
    float mg = 0;
    for( i = 0; i < nivelDeIntensidad; i++ ){
        mg = mg + ( i * p_i[i] );
    }

    return  mg;
}
float varianzaEntreClases( float P1k, float mk, float mg ){
    float varianza = 0;

    varianza = ( ( (mg*P1k) - (mk) ) * ( (mg*P1k) - (mk) ) ) / ( P1k * ( 1 - P1k ) );
    //LOGI("VAlor varianza %d: \n",varianza);
    return varianza;
}

int obtenerUmbralOptimo(float p_i[ ], float varianza, float mg , int kInicial ,int nivelDeIntensidad){
    int k, kOtsu = kInicial;
    float P1k, mk, kResultado;
    for( k = 1; k < nivelDeIntensidad; k++ ){
        P1k = obtenerP1k( k, p_i );
        mk = obtenermk( k, p_i );
        kResultado = varianzaEntreClases( P1k, mk, mg );

        if ( kResultado > varianza ){
            varianza = kResultado;
            kOtsu=k;
        }
    }
    LOGI("VAlor umbral op %d: \n",kOtsu);
    return kOtsu;
}

void binarizarMatriz( Mat imagen1, int kOtsu, Mat& out){

    threshold(imagen1,out,kOtsu,255,THRESH_BINARY);
}

Mat watershedSegment(Mat & image, int & noOfSegments)
{
    //To store the gray version of the image
    Mat gray;
    //To store the thresholded image
    Mat ret;
    //convert the image to grayscale
    cvtColor(image,gray,CV_BGR2GRAY);
    //imshow("Gray Image",gray);

    double otsu = (double)getTickCount();
    //threshold the image
    threshold(gray,ret,0,255,CV_THRESH_BINARY_INV+CV_THRESH_OTSU);
    //metodoOtsu1Umbral(gray,1,ret);
    otsu = (double)getTickCount() - otsu;
    LOGI("execution time otsu = %gms\n", otsu*1000./getTickFrequency());

    double open = (double)getTickCount();
    //Execute morphological-open
    morphologyEx(ret,ret,MORPH_OPEN,Mat::ones(9,9,CV_8SC1),Point(4,4),2);
    open= (double)getTickCount() - open;
    LOGI("execution time morphological-open = %gms\n", open*1000./getTickFrequency());

    double dis1 = (double)getTickCount();
    //get the distance transformation
    Mat distTransformed(ret.rows,ret.cols,CV_32FC1);
    distanceTransform(ret,distTransformed,CV_DIST_L2,3);
    //normalize the transformed image in order to display
    normalize(distTransformed, distTransformed, 0.0, 1, NORM_MINMAX);
    dis1= (double)getTickCount() - dis1;
    LOGI("execution time transformada1 = %gms\n", dis1*1000./getTickFrequency());

    double dis2 = (double)getTickCount();
    //threshold the transformed image to obtain markers for watershed
    threshold(distTransformed,distTransformed,0.1,1,CV_THRESH_BINARY);
    //Renormalize to 0-255 to further calculations
    normalize(distTransformed, distTransformed, 0.0, 255.0, NORM_MINMAX);
    distTransformed.convertTo(distTransformed,CV_8UC1);
    dis2= (double)getTickCount() - dis2;
    LOGI("execution time transformada12 = %gms\n", dis2*1000./getTickFrequency());


    double mk = (double)getTickCount();
    //calculate the contours of markers
    int i, j, compCount = 0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(distTransformed, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    mk= (double)getTickCount() - mk;
    LOGI("execution time marker = %gms\n", mk*1000./getTickFrequency());

    if( contours.empty() )
        return Mat();
    Mat markers(distTransformed.size(), CV_32S);
    markers = Scalar::all(0);
    int idx = 0;
    //draw contours
    for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
        drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

    if( compCount == 0 )
        return Mat();

    //calculate the time taken to the watershed algorithm
    double t = (double)getTickCount();
    //apply watershed with the markers as seeds
    watershed( image, markers );
    t = (double)getTickCount() - t;
    LOGI("execution time watershed = %gms\n", t*1000./getTickFrequency());
    //printf( "execution time = %gms\n", t*1000./getTickFrequency() );

    //create displayable image of segments
    Mat wshed = createSegmentationDisplay(markers,compCount,image);

    //imshow( "watershed transform", wshed );
    noOfSegments = compCount;

    //returns the segments
    return markers;
}

Mat createSegmentationDisplay(Mat & segments,int numOfSegments,Mat & image)
{
    double t = (double)getTickCount();
    //create a new image
    Mat wshed(segments.size(), CV_8UC3);

    //Create color tab for coloring the segments
    vector<Vec3b> colorTab;
    for(int i = 0; i < numOfSegments; i++ )
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    //assign different color to different segments
    for(int i = 0; i < segments.rows; i++ )
    {
        for(int j = 0; j < segments.cols; j++ )
        {
            int index = segments.at<int>(i,j);
            if( index == -1 )
                wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
            else if( index <= 0 || index > numOfSegments )
                wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
            else
                wshed.at<Vec3b>(i,j) = colorTab[index - 1];
        }
    }

    //If the original image available then merge with the colors of segments
    if(image.dims>0) wshed = wshed*0.5+image*0.5;


    return wshed;
    t = (double)getTickCount() - t;
    LOGI("execution time watershed pintado = %gms\n", t*1000./getTickFrequency());
}