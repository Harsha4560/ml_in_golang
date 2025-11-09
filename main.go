package main

import (
	"fmt"
)

func main() {
	path := "test.csv"
	df, err := ReadCsv(path)
	if err != nil {
		fmt.Println("Error: ", err)
	}
	X, _ := NewTensor(len(df["colx"]))
	X.data = df["colx"]
	Y, _ := NewTensor(len(df["coly"]))
	Y.data = df["coly"]
	X, _ = X.Unsqeeze(1)
	Y, _ = Y.Unsqeeze(1)
	LR := NewLinearRegression(1)
	// LR.Weights.Set(1.0, 0)
	// LR.Bias = 1.0
	x, _ := X.Slice(6)
	k, _ := LR.Predict(x)
	fmt.Println("\nThe value before training")
	k.Show()
	fmt.Println()
	err = LR.Train(X, Y, 0.0001, 10000)
	if err != nil {
		fmt.Println("Error: ", err)
	}
	// LR.Weights.Show()
	k, _ = LR.Predict(x)
	fmt.Println("Value after training: ")
	k.Show()
}