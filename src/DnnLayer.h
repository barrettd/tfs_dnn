// --------------------------------------------------------------------
//  DnnLayer.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef dnnLayer_h
#define dnnLayer_h

#include <vector>
#include "Constants.h"

namespace tfs {
    
    class DnnLayer {            // Base class of all layers.
    protected:
        const char  *m_name;    // Used in serialization.
        DNN_NUMERIC *m_w;       // Weights
        DNN_NUMERIC *m_dw;      // Weight derivative
        
        unsigned long m_x;      // Width
        unsigned long m_y;      // Height
        unsigned long m_z;      // Depth
        unsigned long m_size;   // = x * w * h;  // Count of DNN_NUMERIC elements.
        
        DnnLayer *m_prev_layer;
        DnnLayer *m_next_layer;
        
        void setup( unsigned long xx, unsigned long yy, unsigned long zz );
        void teardown( void );
        
    public:
        DnnLayer( const char *name );
        DnnLayer( const char *name, DnnLayer *previousLayer );
        virtual ~DnnLayer( void );
        
        const char*   name(   void ) const;
        unsigned long width(  void ) const; // x
        unsigned long height( void ) const; // y
        unsigned long depth(  void ) const; // z
        unsigned long size(   void ) const; // = x * w * h;  // Count of DNN_NUMERIC elements.
        DNN_NUMERIC* w(  void );            // weights
        DNN_NUMERIC* dw( void );            // weight derivatives
        
        DnnLayer *getPreviousLayer( void ) const;
        DnnLayer *setPreviousLayer( DnnLayer *layer );
        DnnLayer *getNextLayer( void ) const;
        DnnLayer *setNextLayer( DnnLayer *layer );
        
        virtual void randomize( void );     // Randomize weights and bias.

        virtual bool forward( const DNN_NUMERIC *data, const unsigned long length );  // Forward propagate while training
        
        virtual bool backprop( void );  // Back propagate while training
        
        virtual bool predict( const DNN_NUMERIC *data, const unsigned long length );   // Forward progagate when predicting
        
    };
    
    
}   // namespace tfs

#endif /* dnnLayers_h */
