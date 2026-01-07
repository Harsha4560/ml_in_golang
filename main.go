package main

import (
	"fmt"
	"nnscratch/layers"
	"nnscratch/utils"

	"nnscratch/maths"
	"nnscratch/optim"
	"nnscratch/tensor"
	// "nnscratch/utils"
)



func main() {
	df, err := utils.ReadCsv("test.csv")
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
		panic(ok)
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
	model.RandomInitialization()


	epochs := 10000
	batchSize := 2
	loader := utils.NewDataLoader(x, y, batchSize, true)

	optimizer := optim.NewAdam(model.GetParameters(), 0.1)

	for i := 0; i < epochs; i++ {
		sumloss := 0.0
		iterator := loader.MakeIterator()
		for iterator.Next() {
			optimizer.ZeroGrad()
			xBatch, yBatch := iterator.Get()
			prediction, err := model.Forward(xBatch)
			if err != nil {
				panic(err)
			}
			loss, err := model.LossLayer.Loss(yBatch, prediction)
			sumloss += loss
			if err != nil {
				panic(err)
			}

			diff, _ := model.LossLayer.Diffrential()
			err = model.Backward(diff, 0)
			if err != nil {
				panic(err)
			}
			optimizer.Step()
		}
		sumloss = sumloss/float64(batchSize)
		if i%100 == 0 {
			fmt.Println("Epoch: ", i, "Loss: ", sumloss)
			println("-----")
		}

	}
	fmt.Println("training done!")
	ans, _ := model.Forward(x)
	ans, _ = ans.Apply(maths.Round)
	// ans, _ = ans.Apply("Round")
	ans.Show()
}
