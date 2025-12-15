package loss

import (
	"fmt"
	"nnscratch/tensor"
	"nnscratch/maths"

)

// Mean square error function
func MSE(a *tensor.Tensor, b *tensor.Tensor) (float64, error) {
	if !tensor.ShapesMatch(a, b) {
		return 0, fmt.Errorf("mse: The shapes of the tensors do not match %v, %v", a.Shape(), b.Shape())
	}
	temp, err := tensor.TensorDiff(a, b)
	if err != nil {
		return 0, nil
	}
	temp, err = temp.Power(2)
	if err != nil {
		return 0, err
	}
	result, err := temp.Sum()
	if err != nil {
		return 0, err
	}
	result = result / float64(a.Len())
	return result, nil
}

// Diffrention of the MSE function
func DiffMSE(a *tensor.Tensor, b *tensor.Tensor) (*tensor.Tensor, error) {
	if !tensor.ShapesMatch(a, b) {
		return nil, fmt.Errorf("diffMse: The shapes of the tensors do not match %v, %v", a.Shape(), b.Shape())
	}
	result, err := tensor.TensorDiff(a, b)
	if err != nil {
		return nil, err
	}
	result, err = result.MulScalar(2.0 / float64(a.Len()))
	if err != nil {
		return nil, err
	}
	return result, nil
}

// The binary cross entropy function
func BCE(a *tensor.Tensor, b *tensor.Tensor) (float64, error) {
	if !tensor.ShapesMatch(a, b) {
		return 0, fmt.Errorf("bce: The shape of the input tensors do not match")
	}

	ln_res, err := b.Apply(maths.Ln)
	if err != nil {
		return 0, err
	}
	temp1, err := tensor.TensorMul(a, ln_res)
	if err != nil {
		return 0, err
	}
	t_ones, _ := tensor.NewTensorOnes(b.Shape()...)
	ln_res, _ = tensor.TensorDiff(t_ones, b)
	ln_res, _ = ln_res.Apply(maths.Ln)
	a_ones, _ := tensor.TensorDiff(t_ones, a)
	temp2, err := tensor.TensorMul(a_ones, ln_res)
	if err != nil {
		return 0, err
	}
	temp1, _ = tensor.TensorAdd(temp1, temp2)
	result, _ := temp1.Sum()

	result = -1.0 * result / float64(a.Len())
	return result, nil
}

func DiffBCE(a *tensor.Tensor, b *tensor.Tensor) (*tensor.Tensor, error) {
	result, err := tensor.NewTensor(a.Shape()...)
	if err != nil {
		return nil, err
	}
	for i := range a.Data() {
		if a.Data()[i] > 1 || b.Data()[i] > 1 {
			return nil, fmt.Errorf("diffBCE: The value in either tensor is more than one")
		}
		y := a.Data()[i]
		y_pred := b.Data()[i]
		if y_pred < 1e-9 {
			y_pred = 1e-9
		} else if y_pred > 1-1e-9 {
			y_pred = 1 - 1e-9
		}
		result.Data()[i] = (y_pred - y) / (y_pred * (1 - y_pred))
	}
	return result, nil
}
