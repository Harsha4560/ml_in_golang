package layers 

import (
	"nnscratch/tensor"
	"nnscratch/loss"
)

type LossLayer interface {
	Loss(y_pred *tensor.Tensor, y_actual *tensor.Tensor) (float64, error)
	Diffrential() (*tensor.Tensor, error)
}

// Mean square error loss
type MSELossLayer struct {
	y_pred   *tensor.Tensor
	y_actual *tensor.Tensor
}

func (l *MSELossLayer) Loss(y_pred *tensor.Tensor, y_actual *tensor.Tensor) (float64, error) {
	l.y_actual = y_actual
	l.y_pred = y_pred
	res, err := loss.MSE(y_pred, y_actual)
	if err != nil {
		return 0, err
	}
	return res, nil
}

func (l *MSELossLayer) Diffrential() (*tensor.Tensor, error) {
	res, err := loss.DiffMSE(l.y_pred, l.y_actual)
	if err != nil {
		return nil, err
	}
	return res, nil
}


// BCE loss layer should always have a sigmoid function before it to make sure the input is less than 1
type BCELossLayer struct {
	y_pred   *tensor.Tensor
	y_actual *tensor.Tensor
}

func (l *BCELossLayer) Loss(y_pred *tensor.Tensor, y_actual *tensor.Tensor) (float64, error) {
	l.y_actual = y_actual
	l.y_pred = y_pred
	res, err := loss.BCE(y_pred, y_actual)
	if err != nil {
		return 0, err
	}
	return res, nil
}

func (l *BCELossLayer) Diffrential() (*tensor.Tensor, error) {
	res, err := loss.DiffBCE(l.y_pred, l.y_actual)
	if err != nil {
		return nil, err
	}
	return res, nil
}