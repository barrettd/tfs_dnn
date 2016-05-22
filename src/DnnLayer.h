// --------------------------------------------------------------------
//  DnnLayer.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef dnnLayer_h
#define dnnLayer_h

#include "Matrix.h"

namespace tfs {
    
    class DnnLayer {                    // Base class of all layers.
    protected:
        const char *m_name;             // Used in serialization.
        const Matrix *m_in_a;           // Input:    Activations of previous layer, if any.
              Matrix *m_in_dw;          // Input:    dw of previous layer, if any.
        Matrix       *m_w;              // Internal: Weights, to act on input activations from previous layer
        Matrix       *m_dw;             // Internal: Weight derivative, will be null when not training.
        Matrix       *m_out_a;          // Output:   Activations, output of a neuron.
        Matrix       *m_out_dw;         // Output:   Weight derivative, will be null when not training.
        DNN_NUMERIC   m_l1_decay_mul;
        DNN_NUMERIC   m_l2_decay_mul;

        DnnLayer     *m_prev_layer;
        DnnLayer     *m_next_layer;
        
        void setup( const bool trainable = true );
        void teardown( void );

    public:
        DnnLayer( const char *name );
        DnnLayer( const char *name, DnnLayer *previousLayer );
        virtual ~DnnLayer( void );
        
        // Layer attributes:
        const char *name( void ) const;
        Matrix     *outA( void );               // Output Neuron Activations
        Matrix     *weights( void );            // Internal Neuron connection weights    (w)
        Matrix     *gradiants( void );          // Internal Neuron connection gradiants (dw)
        
        DNN_NUMERIC l1DecayMultiplier( void ) const;
        DNN_NUMERIC l1DecayMultiplier( DNN_NUMERIC value );
        
        DNN_NUMERIC l2DecayMultiplier( void ) const;
        DNN_NUMERIC l2DecayMultiplier( DNN_NUMERIC value );
        
        // Linked list pointers:
        DnnLayer *getPreviousLayer( void ) const;
        DnnLayer *setPreviousLayer( DnnLayer *layer );
        DnnLayer *getNextLayer( void ) const;
        DnnLayer *setNextLayer( DnnLayer *layer );
        
        // ----------------------------------------------------------------------------------------------
        // These methods perform a function, then call the next layer in the chain to do the same.
        // ----------------------------------------------------------------------------------------------
        virtual void initialize( void );                // Zero activations, gradiant and randomize weights. Forward calling.
        virtual void randomize(  void );                // Randomize gradiant. Forward calling.
        
        virtual bool forward(  void );                  // Forward propagate while training
        virtual bool backprop( void );                  // Back propagate while training
        virtual DNN_NUMERIC backprop( const  Matrix &expectation );    // Last layer
        virtual DNN_NUMERIC backprop( const DNN_INTEGER expectation );    // Last layer
        
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting
        
    };
    
    
}   // namespace tfs

#endif /* dnnLayers_h */
