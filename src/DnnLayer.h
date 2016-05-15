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
    
    class DnnLayer {                  // Base class of all layers.
    protected:
        const char *m_name;           // Used in serialization.
        const Matrix *m_pa;           // Activations of previous layer, if any.
        Matrix       *m_w;            // Weights, to act on input activations from previous layer
        Matrix       *m_dw;           // Weight derivative, will be null when not training.
        Matrix       *m_a;            // Activations, output of a neuron.
        DnnLayer     *m_prev_layer;
        DnnLayer     *m_next_layer;
        
        void setup( const unsigned long  inX, const unsigned long  inY, const unsigned long  inZ,
                    const unsigned long outX, const unsigned long outY, const unsigned long outZ,
                    const bool trainable = true );
        void setup( const Matrix *activations, const bool trainable = true );
        void teardown( void );

        bool forward( const Matrix &data );     // Forward propagate while training

    public:
        DnnLayer( const char *name );
        DnnLayer( const char *name, DnnLayer *previousLayer );
        virtual ~DnnLayer( void );
        
        // Layer attributes:
        const char *name( void ) const;
        Matrix     *w(  void );             // Weights
        Matrix     *dw( void );             // Weight derivatives
        Matrix     *a(  void );             // Activations
        
        virtual unsigned long aX( void ) const; // Activation dimensions
        virtual unsigned long aY( void ) const;
        virtual unsigned long aZ( void ) const;
        virtual unsigned long aSize( void ) const;
        
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
        
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting
        
    };
    
    
}   // namespace tfs

#endif /* dnnLayers_h */
