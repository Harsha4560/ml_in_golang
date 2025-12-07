package main

import (
	"fmt"
)

func main() {
	df, err := ReadCsv("test.csv")
	if err != nil {
		fmt.Print(err)
	}
	x1 := df["colx1"]
	x2 := df["colx2"]
	x1t, _ := NewTensorInput(x1)
	x2t, _ := NewTensorInput(x2)
	x, err := TensorCombine(x1t, x2t)
	if err != nil {
		fmt.Print(err)
	}
	x, _ = x.Transpose()
	// x.Show()
	y1, ok := df["coly"] // Check existence if needed, or just y1 := df["coly"]
	if !ok {
		// handle error
	}
	y, _ := NewTensorInput(y1)
	y, _ = y.Unsqeeze(1)
	// y.Show()

	model := NewSequential(
		NewDenseLayer(2, 4),
		&SigmoidLayer{},
		NewDenseLayer(4, 1),
		&SigmoidLayer{},
	)
	model.LossLayer = &BCELossLayer{}

	epochs := 100000

	for i := 0; i < epochs; i++ {
		prediction, err := model.Forward(x)
		if err != nil {
			panic(err)
		}
		loss, err := model.LossLayer.Loss(y, prediction)
		if err != nil {
			panic(err)
		}
		if i%10000 == 0 {
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
