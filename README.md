# shaper
 Create a target image from given shapes like triangles, cubes, circles, ...

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
