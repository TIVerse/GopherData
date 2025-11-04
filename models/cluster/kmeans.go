// Package cluster provides clustering algorithms.
package cluster

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	seriesPkg "github.com/TIVerse/GopherData/series"
)

// KMeans implements the K-Means clustering algorithm.
// Partitions data into K clusters by minimizing within-cluster variance.
type KMeans struct {
	// NClusters is the number of clusters (K)
	NClusters int
	
	// MaxIter is the maximum number of iterations
	MaxIter int
	
	// Tol is the convergence tolerance
	Tol float64
	
	// Init specifies initialization method: "k-means++" or "random"
	Init string
	
	// Seed for random number generator
	Seed int64
	
	// centers stores the cluster centroids
	centers [][]float64
	
	// labels stores the cluster assignment for each sample
	labels []int
	
	// inertia is the sum of squared distances to nearest cluster center
	inertia float64
	
	// fitted indicates whether the model has been fitted
	fitted bool
	
	// nIter stores the number of iterations performed
	nIter int
}

// NewKMeans creates a new K-Means clusterer.
func NewKMeans(nClusters int, maxIter int, init string, seed int64) *KMeans {
	if nClusters < 1 {
		nClusters = 3
	}
	if maxIter <= 0 {
		maxIter = 300
	}
	if init == "" {
		init = "k-means++"
	}
	
	return &KMeans{
		NClusters: nClusters,
		MaxIter:   maxIter,
		Tol:       1e-4,
		Init:      init,
		Seed:      seed,
		fitted:    false,
	}
}

// Fit trains the K-Means model on data X.
func (km *KMeans) Fit(X *dataframe.DataFrame) error {
	// Extract features
	features, err := extractNumericFeatures(X)
	if err != nil {
		return err
	}
	
	n := len(features)
	if n < km.NClusters {
		return fmt.Errorf("n_samples=%d should be >= n_clusters=%d", n, km.NClusters)
	}
	
	p := len(features[0])
	
	// Initialize centers
	rng := rand.New(rand.NewSource(km.Seed))
	
	if km.Init == "k-means++" {
		km.centers = km.initKMeansPlusPlus(features, rng)
	} else {
		km.centers = km.initRandom(features, rng)
	}
	
	km.labels = make([]int, n)
	
	// K-Means iterations
	for iter := 0; iter < km.MaxIter; iter++ {
		// Assignment step: assign each point to nearest center
		changed := 0
		for i, point := range features {
			oldLabel := km.labels[i]
			minDist := math.MaxFloat64
			bestCluster := 0
			
			for j, center := range km.centers {
				dist := euclideanDistance(point, center)
				if dist < minDist {
					minDist = dist
					bestCluster = j
				}
			}
			
			km.labels[i] = bestCluster
			if bestCluster != oldLabel {
				changed++
			}
		}
		
		// Update step: recompute cluster centers
		newCenters := make([][]float64, km.NClusters)
		clusterSizes := make([]int, km.NClusters)
		
		for j := 0; j < km.NClusters; j++ {
			newCenters[j] = make([]float64, p)
		}
		
		for i, point := range features {
			cluster := km.labels[i]
			clusterSizes[cluster]++
			for j := 0; j < p; j++ {
				newCenters[cluster][j] += point[j]
			}
		}
		
		// Average to get new centers
		maxShift := 0.0
		for j := 0; j < km.NClusters; j++ {
			if clusterSizes[j] > 0 {
				for k := 0; k < p; k++ {
					newCenters[j][k] /= float64(clusterSizes[j])
				}
				
				// Calculate shift
				shift := euclideanDistance(km.centers[j], newCenters[j])
				if shift > maxShift {
					maxShift = shift
				}
			}
		}
		
		km.centers = newCenters
		km.nIter = iter + 1
		
		// Check convergence
		if maxShift < km.Tol {
			break
		}
	}
	
	// Calculate inertia (sum of squared distances)
	km.inertia = 0
	for i, point := range features {
		cluster := km.labels[i]
		dist := euclideanDistance(point, km.centers[cluster])
		km.inertia += dist * dist
	}
	
	km.fitted = true
	return nil
}

// Predict assigns cluster labels to new data.
func (km *KMeans) Predict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error) {
	if !km.fitted {
		return nil, fmt.Errorf("model not fitted yet")
	}
	
	features, err := extractNumericFeatures(X)
	if err != nil {
		return nil, err
	}
	
	labels := make([]any, len(features))
	
	for i, point := range features {
		minDist := math.MaxFloat64
		bestCluster := 0
		
		for j, center := range km.centers {
			dist := euclideanDistance(point, center)
			if dist < minDist {
				minDist = dist
				bestCluster = j
			}
		}
		
		labels[i] = int64(bestCluster)
	}
	
	return seriesPkg.New("cluster", labels, core.DtypeInt64), nil
}

// FitPredict fits the model and returns cluster labels.
func (km *KMeans) FitPredict(X *dataframe.DataFrame) (*seriesPkg.Series[any], error) {
	err := km.Fit(X)
	if err != nil {
		return nil, err
	}
	
	labels := make([]any, len(km.labels))
	for i, label := range km.labels {
		labels[i] = int64(label)
	}
	
	return seriesPkg.New("cluster", labels, core.DtypeInt64), nil
}

// Centers returns the cluster centroids.
func (km *KMeans) Centers() [][]float64 {
	return km.centers
}

// Inertia returns the sum of squared distances to cluster centers.
func (km *KMeans) Inertia() float64 {
	return km.inertia
}

// NIter returns the number of iterations performed.
func (km *KMeans) NIter() int {
	return km.nIter
}

// initKMeansPlusPlus initializes centers using K-Means++ algorithm.
func (km *KMeans) initKMeansPlusPlus(features [][]float64, rng *rand.Rand) [][]float64 {
	n := len(features)
	p := len(features[0])
	centers := make([][]float64, km.NClusters)
	
	// Choose first center randomly
	firstIdx := rng.Intn(n)
	centers[0] = make([]float64, p)
	copy(centers[0], features[firstIdx])
	
	// Choose remaining centers
	for i := 1; i < km.NClusters; i++ {
		// Calculate squared distances to nearest center
		distances := make([]float64, n)
		sumDist := 0.0
		
		for j, point := range features {
			minDist := math.MaxFloat64
			for k := 0; k < i; k++ {
				dist := euclideanDistance(point, centers[k])
				if dist < minDist {
					minDist = dist
				}
			}
			distances[j] = minDist * minDist
			sumDist += distances[j]
		}
		
		// Choose next center with probability proportional to squared distance
		r := rng.Float64() * sumDist
		cumSum := 0.0
		nextIdx := 0
		
		for j, dist := range distances {
			cumSum += dist
			if cumSum >= r {
				nextIdx = j
				break
			}
		}
		
		centers[i] = make([]float64, p)
		copy(centers[i], features[nextIdx])
	}
	
	return centers
}

// initRandom initializes centers randomly from data points.
func (km *KMeans) initRandom(features [][]float64, rng *rand.Rand) [][]float64 {
	n := len(features)
	p := len(features[0])
	centers := make([][]float64, km.NClusters)
	
	// Choose K random points as initial centers
	indices := rng.Perm(n)[:km.NClusters]
	
	for i, idx := range indices {
		centers[i] = make([]float64, p)
		copy(centers[i], features[idx])
	}
	
	return centers
}

// Helper functions

func extractNumericFeatures(X *dataframe.DataFrame) ([][]float64, error) {
	n := X.Nrows()
	cols := X.Columns()
	
	// Filter numeric columns
	numericCols := make([]string, 0)
	for _, col := range cols {
		series, err := X.Column(col)
		if err != nil {
			continue
		}
		if isNumericDtype(series.Dtype()) {
			numericCols = append(numericCols, col)
		}
	}
	
	if len(numericCols) == 0 {
		return nil, fmt.Errorf("no numeric columns found")
	}
	
	features := make([][]float64, n)
	for i := range features {
		features[i] = make([]float64, len(numericCols))
	}
	
	for j, col := range numericCols {
		series, _ := X.Column(col)
		for i := 0; i < n; i++ {
			val, ok := series.Get(i)
			if ok && val != nil {
				features[i][j] = toFloat64Cluster(val)
			}
		}
	}
	
	return features, nil
}

func isNumericDtype(dtype core.Dtype) bool {
	return dtype == core.DtypeFloat64 || dtype == core.DtypeInt64
}

func toFloat64Cluster(val any) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	default:
		return 0
	}
}

func euclideanDistance(p1, p2 []float64) float64 {
	if len(p1) != len(p2) {
		return math.MaxFloat64
	}
	
	sum := 0.0
	for i := range p1 {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}
	
	return math.Sqrt(sum)
}
