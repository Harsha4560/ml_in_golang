package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

// mean square error function
func MSE(a *Tensor, b *Tensor) (float64, error) {
	if !ShapesMatch(a.shape, b.shape) {
		return 0, fmt.Errorf("mse: The shapes of the tensors do not match %v, %v", a.shape, b.shape)
	}
	result := 0.0
	for i := range a.data {
		diff := a.data[i] - b.data[i]
		square, err := Power(diff, 2)
		if err != nil {
			// handle error
			return 0, err
		}
		result += square
	}
	result = result / float64(len(a.data))
	return result, nil
}

// Diffrention of the MSE function
func DiffMSE(a *Tensor, b *Tensor) (*Tensor, error) {
	if !ShapesMatch(a.shape, b.shape) {
		return nil, fmt.Errorf("diffMse: The shapes of the tensors do not match %v, %v", a.shape, b.shape)
	}
	result, err := NewTensor(a.shape...)
	if err != nil {
		return nil, err
	}
	for i := range a.data {
		diff := a.data[i] - b.data[i]
		result.data[i] = diff
	}
	result, err = result.MulScalar(2.0/float64(len(a.data)))
	if err != nil {
		return nil, err
	}
	return result, nil 
}

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
		for i, value := range(row) {
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