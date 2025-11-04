package features

import (
	"fmt"
	"sync"

	"github.com/TIVerse/GopherData/dataframe"
)

// PipelineStep represents a single transformation step in a pipeline.
type PipelineStep struct {
	Name      string
	Estimator Estimator
	fitted    bool
}

// Pipeline chains multiple transformers in sequence.
// Each step is fitted and applied in order.
type Pipeline struct {
	steps []PipelineStep
	mu    sync.RWMutex
}

// NewPipeline creates a new empty pipeline.
func NewPipeline() *Pipeline {
	return &Pipeline{
		steps: make([]PipelineStep, 0),
	}
}

// Add appends a transformer to the pipeline.
// Returns the pipeline for method chaining.
func (p *Pipeline) Add(name string, estimator Estimator) *Pipeline {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.steps = append(p.steps, PipelineStep{
		Name:      name,
		Estimator: estimator,
		fitted:    false,
	})
	return p
}

// Steps returns a copy of the pipeline steps.
func (p *Pipeline) Steps() []PipelineStep {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	steps := make([]PipelineStep, len(p.steps))
	copy(steps, p.steps)
	return steps
}

// Fit trains all transformers in the pipeline.
// Each transformer is fitted on the output of the previous transformer.
func (p *Pipeline) Fit(df *dataframe.DataFrame, target ...string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if len(p.steps) == 0 {
		return fmt.Errorf("pipeline is empty")
	}
	
	current := df
	for i := range p.steps {
		// Fit the current step
		if err := p.steps[i].Estimator.Fit(current, target...); err != nil {
			return fmt.Errorf("step %q (index %d): %w", p.steps[i].Name, i, err)
		}
		p.steps[i].fitted = true
		
		// Transform for the next step
		transformed, err := p.steps[i].Estimator.Transform(current)
		if err != nil {
			return fmt.Errorf("step %q (index %d): %w", p.steps[i].Name, i, err)
		}
		current = transformed
	}
	
	return nil
}

// Transform applies all fitted transformers to the data.
// Returns an error if the pipeline hasn't been fitted.
func (p *Pipeline) Transform(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	if len(p.steps) == 0 {
		return df, nil
	}
	
	current := df
	for i, step := range p.steps {
		if !step.fitted {
			return nil, fmt.Errorf("pipeline not fitted: step %q (index %d) not fitted", step.Name, i)
		}
		
		transformed, err := step.Estimator.Transform(current)
		if err != nil {
			return nil, fmt.Errorf("step %q (index %d): %w", step.Name, i, err)
		}
		current = transformed
	}
	
	return current, nil
}

// FitTransform fits the pipeline and transforms the data in one operation.
func (p *Pipeline) FitTransform(df *dataframe.DataFrame, target ...string) (*dataframe.DataFrame, error) {
	if err := p.Fit(df, target...); err != nil {
		return nil, err
	}
	return p.Transform(df)
}

// IsFitted returns true if all steps in the pipeline are fitted.
func (p *Pipeline) IsFitted() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	for _, step := range p.steps {
		if !step.fitted {
			return false
		}
	}
	return len(p.steps) > 0
}

// Len returns the number of steps in the pipeline.
func (p *Pipeline) Len() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.steps)
}

// GetStep returns the step at the given index.
// Returns nil if the index is out of bounds.
func (p *Pipeline) GetStep(index int) *PipelineStep {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	if index < 0 || index >= len(p.steps) {
		return nil
	}
	
	step := p.steps[index]
	return &step
}

// GetStepByName returns the first step with the given name.
// Returns nil if no step with that name exists.
func (p *Pipeline) GetStepByName(name string) *PipelineStep {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	for _, step := range p.steps {
		if step.Name == name {
			return &step
		}
	}
	return nil
}
