# Seam Carving

This is a program that takes in an image and resizes it using Seam Carving. It only has support for removing seams, but it can use both backward and forward energy methods.

## Usage

`python3 seam.py [-h] [--vertical] [--forward] <image> <scale>`

## Dependencies

- `python 3.9+`

### Python Dependencies

- `numpy`
- `scipy`
- `imageio`
- `numba`

## Results

### Horizontal Resizing

#### Original

![](https://github.com/is386/SeamCarving/blob/main/images/castle.jpg?raw=true)

#### 50% Scale

![](https://github.com/is386/SeamCarving/blob/main/images/castle_out.jpg?raw=true)

### Vertical Resizing

#### Original

![](https://github.com/is386/SeamCarving/blob/main/images/night.jpg?raw=true)

#### 50% Scale

![](https://github.com/is386/SeamCarving/blob/main/images/night_out.jpg?raw=true)

### Forward vs Backward Energy

#### Original

![](https://github.com/is386/SeamCarving/blob/main/images/pliers.jpg?raw=true)

#### Backward

![](https://github.com/is386/SeamCarving/blob/main/images/pliers_out1.jpg?raw=true)

#### Forward

![](https://github.com/is386/SeamCarving/blob/main/images/pliers_out2.jpg?raw=true)
