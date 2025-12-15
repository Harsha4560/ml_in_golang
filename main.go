package main

import (
	"fmt"
	"nnscratch/tensor"
	"nnscratch/layers"
	"strconv"
	"encoding/csv"
	"os"

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


func main() {
	df, err := ReadCsv("test.csv")
	if err != nil {
		fmt.Print(err)
	}
	x1 := df["colx1"]
	x2 := df["colx2"]
	x1t, _ := tensor.NewTensorInput(x1)
	x2t, _ := tensor.NewTensorInput(x2)
	x, err := tensor.TensorCombine(x1t, x2t)
	if err != nil {
		fmt.Print(err)
	}
	x, _ = x.Transpose()
	// x.Show()
	y1, ok := df["coly"] // Check existence if needed, or just y1 := df["coly"]
	if !ok {
		// handle error
	}
	y, _ := tensor.NewTensorInput(y1)
	y, _ = y.Unsqeeze(1)
	// y.Show()

	model := layers.NewSequential(
		layers.NewDenseLayer(2, 4),
		&layers.SigmoidLayer{},
		layers.NewDenseLayer(4, 1),
		&layers.SigmoidLayer{},
	)
	model.LossLayer = &layers.BCELossLayer{}

	epochs := 10000

	for i := 0; i < epochs; i++ {
		prediction, err := model.Forward(x)
		if err != nil {
			panic(err)
		}
		loss, err := model.LossLayer.Loss(y, prediction)
		if err != nil {
			panic(err)
		}
		if i%100 == 0 {
			fmt.Println("Epoch: ", i, "Loss: ", loss)
			println("-----")
		}

		diff, err := model.LossLayer.Diffrential()
		if err != nil {
			panic(err)
		}
		err = model.Backward(diff, 0.1)
		if err != nil {
			panic(err)
		}
	}
	fmt.Println("training done!")
	ans, _ := model.Forward(x)
	// ans, _ = ans.Apply("Round")
	ans.Show()
}
