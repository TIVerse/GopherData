package main

import (
	"fmt"
	"log"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/io/csv"
	"github.com/TIVerse/GopherData/series"
)

func main() {
	fmt.Println("=== GopherData Phase 1 Example ===")
	fmt.Println()

	// Example 1: Create DataFrame from map
	fmt.Println("1. Creating DataFrame from map...")
	df, err := dataframe.New(map[string]any{
		"name":   []string{"Alice", "Bob", "Charlie", "Diana"},
		"age":    []int64{25, 30, 35, 28},
		"salary": []float64{50000, 65000, 75000, 58000},
		"active": []bool{true, true, false, true},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(df.Head(10))

	// Example 2: Select columns
	fmt.Println("\n2. Selecting columns (name, age)...")
	subset := df.Select("name", "age")
	fmt.Println(subset.Head(10))

	// Example 3: Filter rows
	fmt.Println("\n3. Filtering rows (age > 28)...")
	filtered := df.Filter(func(r *dataframe.Row) bool {
		age, ok := r.Get("age")
		if !ok {
			return false
		}
		return age.(int64) > 28
	})
	fmt.Println(filtered.Head(10))

	// Example 4: Series operations
	fmt.Println("\n4. Series operations...")
	_, err = df.Column("age")
	if err != nil {
		log.Fatal(err)
	}

	// Create numeric series for operations
	ageInt64 := series.New("age", []int64{25, 30, 35, 28}, core.DtypeInt64)
	mean := series.Mean(ageInt64)
	sum := series.Sum(ageInt64)
	min, _ := series.Min(ageInt64)
	max, _ := series.Max(ageInt64)

	fmt.Printf("Age statistics:\n")
	fmt.Printf("  Mean: %.2f\n", mean)
	fmt.Printf("  Sum: %d\n", sum)
	fmt.Printf("  Min: %d\n", min)
	fmt.Printf("  Max: %d\n", max)

	// Example 5: Null handling
	fmt.Println("\n5. Null handling...")
	seriesWithNulls := series.New("scores", []float64{85.5, 90.0, 75.5, 88.0, 92.5}, core.DtypeFloat64)
	seriesWithNulls.SetNull(1) // Mark index 1 as null
	seriesWithNulls.SetNull(3) // Mark index 3 as null

	fmt.Printf("Original series:\n%s\n", seriesWithNulls.String())
	fmt.Printf("Null count: %d\n", seriesWithNulls.NullCount())

	meanScores := series.Mean(seriesWithNulls)
	fmt.Printf("Mean (skipping nulls): %.2f\n", meanScores)

	filled := seriesWithNulls.FillNA(80.0)
	fmt.Printf("\nAfter FillNA(80.0):\n%s\n", filled.String())

	dropped := seriesWithNulls.DropNA()
	fmt.Printf("After DropNA():\n%s\n", dropped.String())

	// Example 6: Write and read CSV
	fmt.Println("\n6. CSV I/O...")
	
	// Create a sample CSV
	csvData, err := dataframe.New(map[string]any{
		"product": []string{"Laptop", "Mouse", "Keyboard", "Monitor"},
		"price":   []float64{999.99, 25.50, 75.00, 299.99},
		"stock":   []int64{15, 150, 80, 25},
	})
	if err != nil {
		log.Fatal(err)
	}

	// Write to CSV
	err = csv.WriteCSV(csvData, "products.csv")
	if err != nil {
		log.Fatal("Failed to write CSV:", err)
	}
	fmt.Println("✓ Written to products.csv")

	// Read from CSV
	readDf, err := csv.ReadCSV("products.csv")
	if err != nil {
		log.Fatal("Failed to read CSV:", err)
	}
	fmt.Println("✓ Read from products.csv")
	fmt.Println(readDf.Head(10))

	fmt.Println("\n=== Phase 1 Implementation Complete! ===")
	fmt.Println("\n✅ Core types and interfaces")
	fmt.Println("✅ BitSet for null masks")
	fmt.Println("✅ Generic Series with null handling")
	fmt.Println("✅ DataFrame with selection and filtering")
	fmt.Println("✅ CSV reader/writer with type inference")
}
