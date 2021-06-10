package com.example.trabimg;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ExifInterface;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int My_PERMISSIONS_REQUEST_READ_EXTERNAL=1;
    private static final int My_PERMISSIONS_REQUEST_WRITE_EXTERNAL=1;
    Mat inputImage;
    Mat otsuOut = new Mat();
    Mat mio = new Mat();
    Bitmap bitmap;
    Bitmap resultImg;
    int[] resultInt;

    static {
        if (OpenCVLoader.initDebug()){
            Log.i(TAG,"opencv cargado");
        }else{
            Log.i(TAG,"opencv no cargado");
        }
    }


    static {
        System.loadLibrary("MyOpencvLibs");
    }

    //funciones de Native class
    public static native int[] grayProc(int[] pixels, int w, int h);
    public native void watershed (String filename, long otsu);

    ImageView img1;
    Button abrir;
    Button gray;
    Button fin;

    final String FILENAME = "/storage/emulated/0/bread1400x770.png";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        OpenCVLoader.initDebug();
        checkPermissionW();
        checkPermission();

        //inicializa variables visibles

        img1 = (ImageView)findViewById(R.id.imageView);
        abrir = (Button) findViewById(R.id.btn_abrir);
        gray = (Button) findViewById(R.id.btn_gray);
        fin = (Button) findViewById(R.id.btn_final);

        abrir.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                bitmap= BitmapFactory.decodeFile(FILENAME);//reading the image from drawable
                img1.setImageBitmap(bitmap);// display the decoded image
            }
        });

        gray.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int w = bitmap.getWidth();
                int h = bitmap.getHeight();
                int[] pixels = new int[w*h];
                bitmap.getPixels(pixels, 0, w, 0, 0, w, h);
                resultInt = grayProc(pixels, w, h);
                resultImg = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                resultImg.setPixels(resultInt, 0, w, 0, 0, w, h);
                img1.setImageBitmap(resultImg);
            }
        });

        fin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ot();
                //Utils.bitmapToMat(resultImg,mio);
                String filename = ("/storage/emulated/0/otsu.png");

                Boolean bool = Imgcodecs.imwrite(filename, otsuOut);

                if (bool)
                    Log.i(TAG, "SUCCESS otsu");
                else
                    Log.i(TAG, "Fail otsu");

                Bitmap bitmap2= BitmapFactory.decodeFile("/storage/emulated/0/otsu.png");//reading the image from drawable
                img1.setImageBitmap(bitmap2);// display the decoded image

            }
        });
    }

    public Mat ot(){
        watershed("/storage/emulated/0/bread1400x770.png",otsuOut.getNativeObjAddr());
        return otsuOut;
    }

    private void checkPermission() {
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED){
            if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this,
                    Manifest.permission.READ_EXTERNAL_STORAGE)){
            }else{
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}
                        ,My_PERMISSIONS_REQUEST_READ_EXTERNAL);
            }

        }

    }
    private void checkPermissionW() {


        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED){
            if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)){
            }else{
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}
                        ,My_PERMISSIONS_REQUEST_WRITE_EXTERNAL);
            }

        }
    }


}
