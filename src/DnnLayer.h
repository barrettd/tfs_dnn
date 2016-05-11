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
    
    class DnnLayer {                // Base class of all layers.
    protected:
        const char *m_name;         // Used in serialization.
        Matrix     *m_w;            // Weights
        Matrix     *m_dw;           // Weight derivative, will be null when not training.
        Matrix     *m_a;            // Activations, output of a neuron.
        DnnLayer   *m_prev_layer;
        DnnLayer   *m_next_layer;
        
        void setup( unsigned long xx, unsigned long yy, unsigned long zz, bool training = true );
        void teardown( void );
        
    public:
        DnnLayer( const char *name );
        DnnLayer( const char *name, DnnLayer *previousLayer );
        virtual ~DnnLayer( void );
        
        const char *name(   void ) const;
        Matrix     *w(  void );             // Weights
        Matrix     *dw( void );             // Weight derivatives
        Matrix     *a(  void );             // Activations
        
        DnnLayer *getPreviousLayer( void ) const;
        DnnLayer *setPreviousLayer( DnnLayer *layer );
        DnnLayer *getNextLayer( void ) const;
        DnnLayer *setNextLayer( DnnLayer *layer );
        
        virtual void initialize( void );                // Zero activations, gradiant and randomize weights.
        virtual void randomize(  void );                // Randomize gradiant.
        
        virtual bool forward( const Matrix &data );     // Forward propagate while training
        virtual bool backprop( void );                  // Back propagate while training
        
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting
        
    };
    
    
}   // namespace tfs

#endif /* dnnLayers_h */
