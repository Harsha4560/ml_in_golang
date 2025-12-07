package main

import (
	"fmt"
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

// Gives sum of all the values in the tensor
func (t *Tensor) Sum() (*Tensor, error) {
	result, _ := NewTensor(1)
	for i := range(t.data) {
		result.data[0] += t.data[i]
	}
	return result, nil
}

// Get the value at the given coords in the tensor
func (t *Tensor) Get(coords ...int) (float64, error) {
	index, err := t.getIndex(coords...)
	if err != nil {
		return 0, err
	}
	return t.data[index], nil
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

// Check if the shapes of the given tensors match
func ShapesMatch(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Add scalar to all the values in the tensor
func (t *Tensor) AddScalar(other float64) (*Tensor, error) {
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}
	for i := range result.data {
		result.data[i] = t.data[i] + other
	}
	return result, nil
}

// Multiply all the values with the given value in the tensor
func (t *Tensor) MulScalar(other float64) (*Tensor, error) {
	result, err := NewTensor(t.shape...)
	if err != nil {
		return nil, err
	}
	for i := range result.data {
		result.data[i] = t.data[i] * other
	}
	return result, nil
}

// Perform element wise addition for tensors of same shape
func TensorAdd(a *Tensor, b *Tensor) (*Tensor, error) {
	if !ShapesMatch(a.shape, b.shape) {
		return nil, fmt.Errorf("add: tensor shapes do not match %v and %v", a.shape, b.shape)
	}
	result, _ := NewTensor(a.shape...)
	for i := 0; i < a.size; i++ {
		result.data[i] = a.data[i] + b.data[i]
	}
	return result, nil
}

func TensorDiff(a *Tensor, b *Tensor) (*Tensor, error) {
	if !ShapesMatch(a.shape, b.shape) {
		return nil, fmt.Errorf("tensorDiff: tensor shapes do not match %v, %v", a.shape, b.shape)
	}
	result, _ := NewTensor(a.shape...)
	for i:=0; i<a.size; i++ {
		result.data[i] = a.data[i] - b.data[i]
	}
	return result, nil
}

// Element wise multiplication of tensors of the same shape
func  TensorMul(a *Tensor, b *Tensor) (*Tensor, error) {
	if !ShapesMatch(a.shape, b.shape) {
		return nil, fmt.Errorf("tensorMul: tensor shapes do not match %v and %v", a.shape, b.shape)
	}
	result, _ := NewTensor(a.shape...)
	for i := 0; i < a.size; i++ {
		result.data[i] = a.data[i] * b.data[i]
	}
	return result, nil
}

// Combines two tensors [a, b, c] + [a, b, c] -> [2, a, b, c]
// or can combine [a, b, c, d] + [b, c, d] -> [a+1, b, c, d]
func TensorCombine(a *Tensor, b *Tensor) (*Tensor, error) {
	if !ShapesMatch(a.shape, b.shape) && !ShapesMatch(a.shape[1:], b.shape) {
		return nil, fmt.Errorf("tensorCombine: shape mismatch between %v and %v. Tensors must either have identical shapes or the first tensor must be one rank higher (a stacking of the second)", a.shape, b.shape)
	}
	// When both tensors have the same shape
	if ShapesMatch(a.shape, b.shape) {
		newshape := append([]int{2}, a.shape...)
		result, err := NewTensor(newshape...)
		if err != nil {
			return nil, err
		}
		for i := range(result.data) {
			if i > len(a.data)-1 {
				result.data[i] = b.data[i-len(a.data)]
			} else {
				result.data[i] = a.data[i]
			}
		}
		return result, nil
	}

	// When tensor b is one rank lower than a 
	if ShapesMatch(a.shape[1:], b.shape) {
		newshape := a.shape
		newshape[0] += 1
		result, err := NewTensor(newshape...)
		if err != nil {
			return nil, err
		}
		for i := range(result.data) {
			if i > len(a.data) - 1 {
				result.data[i] = b.data[i-len(a.data)]
			} else {
				result.data[i] = a.data[i]
			}
		}
		return result, nil 
	}
	return nil, fmt.Errorf("for some reason there is an error Good luck finding why")
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

// Matrix multiplication on two teensors look inside for all the kinds or muls
func Matmul(a *Tensor, b *Tensor) (*Tensor, error) {
	// if both are 1D tensors then do a elmentwise multiplication and return a single element tensor (m) * (m) => (1)
	if len(a.shape) == 1 && len(b.shape) == 1 {
		if a.shape[0] != b.shape[0] {
			return nil, fmt.Errorf("the 1d tensors should have the same size they have {%v, %v}", a.shape[0], b.shape[0])
		}
		result, _ := NewTensor(1)
		result.data[0] = 0
		for i := 0; i < a.size; i++ {
			result.data[0] += a.data[i] * b.data[i]
		}
		return result, nil
	}

	// if both are 2d tensors then do matrix multiplication if possible (m, k) * (k, p) => {m, p}
	if len(a.shape) == len(b.shape) && len(a.shape) == 2 {
		if a.shape[1] != b.shape[0] {
			return nil, fmt.Errorf("the 2d tensors should have dims (m, k), (k, p) they are %v, %v", a.shape, b.shape)
		}
		result, _ := NewTensor(a.shape[0], b.shape[1])
		for i := 0; i < a.shape[0]; i++ {
			for j := 0; j < b.shape[1]; j++ {
				val := 0.0
				for k := 0; k < a.shape[1]; k++ {
					val1, _ := a.Get(i, k)
					val2, _ := b.Get(k, j)
					val += val1 * val2
				}
				result.Set(val, i, j)
			}
		}
		return result, nil
	}

	// if 2d and 1d tensor then do matrix multiplication if possible (m, k) * (k) => (m)\
	if len(a.shape) == 2 && len(b.shape) == 1 {
		if a.shape[1] != b.shape[0] {
			return nil, fmt.Errorf("the 2d and 1d tensors should have the dims (m, k), (k) they are %v, %v", a.shape, b.shape)
		}
		result, _ := NewTensor(a.shape[0])
		for i := 0; i < a.shape[0]; i++ {
			val := 0.0
			for j := 0; j < a.shape[1]; j++ {
				val1, _ := a.Get(i, j)
				val2, _ := b.Get(j)
				val += val1 + val2
			}
			result.Set(val, i)
		}
		return result, nil
	}

	// if more than 2d tensors then do matrix multiplication on their 2d slices if possible (..., m, k) * (..., k, p) => (..., m, p) the ... should be same
	if len(a.shape) == len(b.shape) && len(a.shape) > 2 {
		if a.shape[len(a.shape)-2] != b.shape[len(a.shape)-1] {
			return nil, fmt.Errorf("the n'd tensors should have the dims (..., m, k), (..., k, p) they are %v, %v", a.shape, b.shape)
		}
		n := len(a.shape) - 2
		coord := make([]int, n)
		newShape := a.shape
		newShape[n+1] = b.shape[n+1]
		result, _ := NewTensor(newShape...)

		total := 1
		for _, dim := range a.shape[:n] {
			total *= dim
		}

		for i := 0; i < total; i++ {
			newCoord := make([]int, n)
			copy(newCoord, coord)
			for m := 0; m < a.shape[n]; m++ {
				for p := 0; p < b.shape[n+1]; p++ {
					val := 0.0
					for k := 0; k < a.shape[n+1]; k++ {
						arr1 := append(newCoord, m, k)
						arr2 := append(newCoord, k, p)
						val1, _ := a.Get(arr1...)
						val2, _ := b.Get(arr2...)
						val += val1 * val2
					}
					arr3 := append(newCoord, m, p)
					result.Set(val, arr3...)
				}
			}
			for j := n - 1; j >= 0; j-- {
				coord[j]++
				if coord[j] < a.shape[j] {
					break
				}
				coord[j] = 0
			}
		}
		return result, nil
	}
	return nil, fmt.Errorf("something wrong last %v, %v", a.shape, b.shape)
}

// Transpose a tensor [n, m] -> [m, n] 
func (t *Tensor) Transpose() (*Tensor, error) {
	if len(t.shape) != 2 {
		return nil, fmt.Errorf("transpose: Input should be a 2d tensor")
	}
	result, _ := NewTensor(t.shape[1], t.shape[0])
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			val, _ := t.Get(i, j)
			result.Set(val, j, i)
		}
	}
	return result, nil
}

// Reshape a given tensor to the input shape
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	mul1 := 1
	for _, i := range t.shape {
		mul1 *= i
	}
	mul2 := 1
	for _, i := range shape {
		mul2 *= i
	}

	if mul1 != mul2 {
		return nil, fmt.Errorf("reshape: The given shape cannot fit the data in the tensor - %v, %v", mul1, mul2)
	}
	result, err := NewTensor(shape...)
	if err != nil {
		// handle error
		return nil, err
	}
	for i := range result.data {
		result.data[i] = t.data[i]
	}
	return result, nil
}

// Flatten the tensor to a 1d vector
func (t *Tensor) Flatten() (*Tensor, error) {
	mul := 1
	for _, i := range t.shape {
		mul *= i
	}
	result, err := NewTensor(mul)
	if err != nil {
		return nil, err
	}
	copy(result.data, t.data)
	return result, nil
}

// Add 1 to the dimention of the tensor at the input value
func (t *Tensor) Unsqeeze(pos int) (*Tensor, error) {
	firstPart := t.shape[:pos]
	secondPartt := t.shape[pos:]
	newSlice := append(firstPart, 1)
	newSlice = append(newSlice, secondPartt...)
	result, err := t.Reshape(newSlice...)
	if err != nil {
		return nil, err
	}
	return result, nil
}

