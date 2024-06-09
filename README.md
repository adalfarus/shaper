# shaper
 Create a target image from given shapes like triangles, cubes, circles, ... . There are two kinds of **images that work best**, **Highly Detailed Images:** These images have lots of fine details and intricate features. When blurred, the details still hold enough information to create a recognizable image and **Very Simple Images:** These images have clear, distinct shapes and minimal elements. Even when blurred, the simplicity makes them easy to recreate. The **worst image** you can give it is one where **color changes are not small enough to be ignored but still too complex**, making it difficult to use the basic shapes effectively.

<p float="center">
  <img src="/from_image.png" width="41%" />
  <img src="/final_image.png" width="41%" />
</p>

Arguments are the target image path, the shapes directory path, changing the color of the shapes, using random shapes, and finally if it should start at another image.
````bash
py .\V4.py 003.JPG ./shapes True False .\final_image_old.png
````
You can leave out the last parameter if you want to start fresh like this:
````bash
py .\V4.py 8.jpg ./shapes True False
````
