package utils

import (
	"encoding/csv"
	"fmt"
	"image"
	"nnscratch/tensor"
	"os"
	"strconv"
)

// Function to read the csv file returns like pandas df
func ReadCsv(path string) (map[string][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("readCsv: error opening the file path: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	headers, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("readCsv: Error reading the headers: %w", err)
	}

	columnMap := make(map[string][]float64)

	for _, header := range headers {
		columnMap[header] = []float64{}
	}
	for {
		row, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return nil, fmt.Errorf("readCsv: Error reading row: %w", err)
		}
		for i, value := range row {
			if i < len(headers) {
				floatVal, err := strconv.ParseFloat(value, 64)
				if err != nil {
					return nil, fmt.Errorf("readCsv: error converting to float: %w", err)
				}
				columnMap[headers[i]] = append(columnMap[headers[i]], floatVal)
			}
		}
	}
	return columnMap, nil
}

func loadImage(file_path string) (image.Image, error) {
	file, err := os.Open(file_path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return img, nil
}

func ColorImgTo3DTensor(file_path string) (*tensor.Tensor, error) {
	img, err := loadImage(file_path)
	if err != nil {
		return nil, err
	}
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	imgTensor, _ := tensor.NewTensor(height, width, 4)
	for y:=0; y<height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			imgTensor.Set(float64(uint8(r >> 8)), y, x, 0)
			imgTensor.Set(float64(uint8(g >> 8)), y, x, 1)
			imgTensor.Set(float64(uint8(b >> 8)), y, x, 2)
			imgTensor.Set(float64(uint8(a >> 8)), y, x, 3)
		}
	}
	return imgTensor, nil
}

func GreyImgTo2DTensor(file_path string) (*tensor.Tensor, error) {
	img, err := loadImage(file_path)
	if err != nil {
		return nil, err
	}
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	imgTensor, _ := tensor.NewTensor(height, width)
	for y:=0; y<height; y++ {
		if grayImg, ok := img.(*image.Gray); ok {
			for x:=0; x<width; x++ {
				imgTensor.Set(float64(grayImg.GrayAt(x, y).Y), y, x)
			}
		} else {
			for x:=0; x<width; x++ {
				r, g, b, _ := img.At(x, y).RGBA()
				imgTensor.Set(float64(uint8((r>>8 + g>>8 + b>>8)/3)), y, x)
			}
		}
	}
	return imgTensor, nil
}
