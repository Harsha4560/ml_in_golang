package main

import (
	"fmt"
)

func main() {
	df, err := ReadCsv("test.csv")
	if err != nil {
		fmt.Print(err)
	}

}