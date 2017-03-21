# Cousera: [Machine Learning](https://www.coursera.org/learn/machine-learning/)
Taught by [Andrew Ng](https://www.coursera.org/instructor/andrewng)

## About the Course
Taken from the website:

About this course: Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI.

This course provides a broad introduction to machine learning, datamining, and statistical pattern recognition. Topics include: (i) Supervised learning (parametric/non-parametric algorithms, support vector machines, kernels, neural networks). (ii) Unsupervised learning (clustering, dimensionality reduction, recommender systems, deep learning). (iii) Best practices in machine learning (bias/variance theory; innovation process in machine learning and AI). The course will also draw from numerous case studies and applications, so that you'll also learn how to apply learning algorithms to building smart robots (perception, control), text understanding (web search, anti-spam), computer vision, medical informatics, audio, database mining, and other areas.

## Prerequisites

The course is taught in English with subtitles in Spanish, Hindi, Japanese, and Chinese. The language used is Matlab or Octave. *For those without Matlab, Octave is a free alternative with similar syntax.*


### Octave
Follow the [link](https://www.gnu.org/software/octave/) for more details about installation. Once the installation is complete, any additional [packages](https://octave.sourceforge.io/packages.php) (e.g., nnet, statistics, nan) can be installed from the prompt with the following commands
```
pkg install -forge package_name
```
Once installed, the package must be loaded to use

```
pkg load package_name
```
Example code is given below
```
b = [4; 5; 6];      # Define columen vector. Include to semicolon to supress output
A = [1 2 3;
     4 5 6;
     7 8 9];
x = A\b             # Solve system Ax=b
```
A simply plotting example:
```
x = -10:0.1:10; # Create an evenly-spaced vector from -10..10
y = sin (x);    # y is also a vector
plot (x, y);
title ("Simple 2-D Plot");
xlabel ("x");
ylabel ("sin (x)");
```
![Simple 2-D Plot](https://github.com/Hosini/CourseraMachineLearning/tree/master/Supplements/example-plot.svg)

## Authors

* **Brandon Touchet ** - *Initial work* - [PurpleBooth](https://github.com/hosini)
