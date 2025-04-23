package org.firstinspires.ftc.teamcode;

import java.util.ArrayList;
import java.util.List;

import org.openftc.easyopencv.OpenCvPipeline;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Core;

import org.opencv.imgproc.Imgproc;

public class DetectionPipeline extends OpenCvPipeline {
    Mat hsv = new Mat();

    // Thresholding
    //                                           degrees, percent, percent
    public Scalar hsvLowerThreshold = new Scalar(118, 34, 0);
    public Scalar hsvUpperThreshold = new Scalar(280, 100, 59);

    // Convert to 2 degrees, 0-255, 0-255
    Scalar hsvTrueLowerThreshold = new Scalar(hsvLowerThreshold.val[0] / 2, hsvLowerThreshold.val[1] * 255 / 100, hsvLowerThreshold.val[2] * 255 / 100);
    Scalar hsvTrueUpperThreshold = new Scalar(hsvUpperThreshold.val[0] / 2, hsvUpperThreshold.val[1] * 255 / 100, hsvUpperThreshold.val[2] * 255 / 100);
    Mat thresholded = new Mat();
    Mat thresholdedRGB = new Mat();

    // Masking
    Mat masked = new Mat();

    // Edge detection
    public double cannyLowerThreshold = 50f;
    public double cannyUpperThreshold = 60f;

    Mat cannyEdge = new Mat();

    // Contours
    List<MatOfPoint> contours = new ArrayList<>();
    Mat hierarchy = new Mat(); 
    public int Ncontours;

    // Output mat (with color labeling, object detection)
    Mat output = new Mat();

    @Override
    public Mat processFrame(Mat input) {
        output = input;

        // Convert to hsv
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        
        // HSV threshold
        Core.inRange(hsv, hsvTrueLowerThreshold, hsvTrueUpperThreshold, thresholded);

        // Mask original pixel values over only the thresholded values (display/debug only)
        Imgproc.cvtColor(thresholded, thresholdedRGB, Imgproc.COLOR_GRAY2RGBA);
        Core.bitwise_and(input, thresholdedRGB, masked);

        // Canny edge detector
        Imgproc.Canny(masked, cannyEdge, cannyLowerThreshold, cannyUpperThreshold);

        // Find contours
        Imgproc.findContours(cannyEdge, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Ncontours = contours.size();

        // Draw contours
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(output, contours, i, new Scalar(0, 255, 0));
        }

        // Get bounding box

        // return thresholded;
        // return masked;
        // return cannyEdge;
        return output;
    }

}
