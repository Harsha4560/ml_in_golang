package layers

import (
	"nnscratch/utils"
	"nnscratch/maths"
)

// He initialization of the model weights. It is not zero anymore to be used when relu units are present
func (model *Sequential) HeInitialization() {
	for _, layer := range model.Layers {
		layer_details := layer.GetModelDetails()[0]
		name := layer_details.Name
		switch name {
		case "Dense Layer":
			inshape := layer_details.InputParameters
			stdDev := maths.Power(2.0/float64(inshape), 0.5)
			weights := layer.GetWeights()[0]
			utils.GenerateNormalData(weights, 0, stdDev)
		}
	}
}

//Random initialization with std dev of 0.01 
func (model *Sequential) RandomInitialization() {
	for _, w := range model.GetParameters() {
		utils.GenerateNormalData(w.Value, 0, 0.01)
	}
}

