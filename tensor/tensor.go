package tensor

import (
	"fmt"
	"math/rand"
	"nnscratch/maths"
	"reflect"
)

type Tensor struct {
	shape   []int
	data    []float64
	strides []int
	size    int
}

// Takes a multidimentional array and converts it to a tensor
func NewTensorInput(input any) (*Tensor, error) {
	val := reflect.ValueOf(input)
	if val.Kind() != reflect.Slice && val.Kind() != reflect.Array {
		return nil, fmt.Errorf("newTensorInput: expected slice or array got %v", val.Kind())
	}
	var shape []int

	current := val
	for current.Kind() == reflect.Slice || current.Kind() == reflect.Array {
		len := current.Len()
		shape = append(shape, len)
		if len > 0 {
			current = current.Index(0)
		} else {
			break
		}
	}
	
	result, err := NewTensor(shape...)
	if err != nil {
		return nil, err
	}


	for i := range(result.data) {
		coords := make([]int, len(shape))
		tempi := i
		for j := range(coords) {
			coords[j] = tempi/result.strides[j]
			tempi = tempi % result.strides[j]
		}
		current := val
		for _, coord := range(coords) {
			current = current.Index(coord)
		}
		// safe conversion thanks gemini
		switch current.Kind() {
		case reflect.Float32, reflect.Float64: 
			result.data[i] = current.Float()
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			result.data[i] = float64(current.Int())
		default: 
			return nil, fmt.Errorf("newTensorInput: Unsupported element at %v: %v", coords, current.Kind())
		}
	}
	return result, nil
}

// Creates new tensor of the shape with 0 for all the values
func NewTensor(shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("newtensor error: shape of Tensor cannot be empty")
	}
	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("newtensor error: dimensions must be positive :%v", shape)
		}
		size *= dim
	}

	strides := make([]int, len(shape))
	stride := 1

	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor{
		shape:   shape,
		data:    make([]float64, size),
		strides: strides,
		size:    size,
	}, nil
}

// Creates new tensor of the shape with 1 for all the vales
func NewTensorOnes(shape ...int) (*Tensor, error) {
	t, err := NewTensor(shape...)
	if err != nil {
		return nil, err
	}
	for i := range(t.data) {
		t.data[i] = 1.0
	}
	return t, nil
}

func NewTensorRandom(shape ...int) (*Tensor, error) {
	res, err := NewTensor(shape...)
	if err != nil {
		return nil, err
	}
	for i := range res.data {
		res.data[i] = rand.NormFloat64() * 0.01
	}
	return res, nil
}

// gets the 1d index in the data from the n'd coordinates given
func (t *Tensor) getIndex(coords ...int) (int, error) {
	if len(coords) != len(t.shape) {
		return 0, fmt.Errorf("getIndex error: invalid coordinates for the get function")
	}

	index := 0
	for i, coord := range coords {
		if coord < 0 || coord >= t.shape[i] {
			return 0, fmt.Errorf(" getIndex error: coordinate %d out of bound for dimention %d", coord, i)
		}
		index += t.strides[i] * coord
	}
	return index, nil
}

// Set the value of a given coordinates
func (t *Tensor) Set(value float64, coords ...int) error {
	index, err := t.getIndex(coords...)
	if err != nil {
		return err
	}
	t.data[index] = value
	return nil
}

// Sets the sub tensor at the given coords
func (t *Tensor) SetTensor(val *Tensor, coords ...int) error {
	if len(coords) >= len(t.shape) {
		return fmt.Errorf("setTensor: the given tensor has the wrong shape for assigning tensors %v", t.shape)
	}
	for i := 0; i < len(val.shape); i++ {
		if val.shape[len(val.shape)-1-i] != t.shape[len(t.shape)-1-i] {
			return fmt.Errorf("setTensor: the shapes do not match")
		}
	}
	start := 0
	for i := 0; i < len(t.shape)-len(val.shape); i++ {
		start += coords[i] * t.strides[i]
	}
	for i := 0; i < len(val.data); i++ {
		t.data[i+start] = val.data[i]
	}
	return nil
}

// Give the sub tensor at the given coords
func (t *Tensor) Slice(coords ...int) (*Tensor, error) {
	if len(coords) == 0 {
		return nil, fmt.Errorf("slice error: slice must have atleast one coords")
	}
	if len(coords) > len(t.shape) {
		return nil, fmt.Errorf("slice error: too many coordinates for a slice use get for single elements")
	}

	// return the single value as a one element tensor if called with slice
	if len(coords) == len(t.shape) {
		data, err := t.Get(coords...)
		if err != nil {
			return nil, err
		}
		return &Tensor{
			shape:   []int{1},
			data:    []float64{data},
			strides: []int{1},
			size:    1,
		}, nil
	}

	newDim := len(coords)
	newShape := t.shape[newDim:]
	newStrides := t.strides[newDim:]
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	startIndex := 0
	for i := range coords {
		startIndex += coords[i] * t.strides[i]
	}
	if startIndex+newSize > len(t.data) {
		return nil, fmt.Errorf("slice: the given slice %d is out of bounds for the given tensor of %d", startIndex+newSize, len(t.data))
	}
	newData := t.data[startIndex : startIndex+newSize]

	return &Tensor{
		shape:   newShape,
		size:    newSize,
		strides: newStrides,
		data:    newData,
	}, nil
}

// Give a piece of the tensor without changing dim t[m:n]
func (t * Tensor) View(a int, b int) (*Tensor, error) {
	if (a < 0 || b > t.shape[0]) {
		return nil, fmt.Errorf("view: The given inputs are not in range of (0, %v) they are (%v, %v)", t.shape[0], a, b)
	}
	if (a > b) {
		return nil, fmt.Errorf("view: first input should be less than the second input they are %v, %v", a, b)
	}
	NewShape := t.shape
	NewShape[0] = b - a
	res, err := NewTensor(NewShape...)
	if err != nil {
		return nil, err
	}
	for i:=a*t.strides[0]; i<b*t.strides[0]; i++ {
		res.data[i-a*t.strides[0]] = t.data[i]
	}
	return res, nil
}

// Check if the shapes of the given tensors match
func ShapesMatch(a, b *Tensor) bool {
	if len(a.shape) != len(b.shape) {
		return false
	}
	for i := range a.shape {
		if a.shape[i] != b.shape[i] {
			return false
		}
	}
	return true
}

//Give a tensor with all the indices given into a single tensor from the input tensor 
func (t *Tensor) GetBatchElements(indices []int) (*Tensor, error) {
	newShape := append([]int{len(indices)}, t.shape[1:]...)
	result, _ := NewTensor(newShape...)
	for i := range indices {
		temp, err := t.Slice(indices[i])
		if err != nil {
			return nil, err
		}
		result.SetTensor(temp, i)
	}
	return result, nil
}

// Shows a given tensor in the terminal
func (t Tensor) Show() {
	if len(t.shape) == 1 {
		fmt.Printf("%v", t.data)
		return
	}
	fmt.Printf("\n")
	for i := range t.shape[0] {
		slice, _ := t.Slice(i)
		slice.Show()
	}
}

// Copies one tensor to a new variable that it is assigned to
func (t *Tensor) Copy() *Tensor {
	newt, _ := NewTensor(t.shape...)
	copy(newt.data, t.data)
	return newt
}


// Apply func that applies a float to float function on a tensor
func (t *Tensor) Apply(fn func(float64) float64) (*Tensor, error) {
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}
	for i := range t.data {
		result.data[i] = fn(t.data[i])
	}
	return result, nil
}

// Get length of the tensor data 
func (t *Tensor) Len() int {
	return len(t.data)
}

func (t *Tensor) Power(a float64) (*Tensor, error) {
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}
	for i := range t.data {
		result.data[i] = maths.Power(t.data[i], a) 
	}
	return result, nil
}

func (t *Tensor) Shape() []int {
	return t.shape 
}

func (t *Tensor) Data() []float64 {
	return t.data
}
