// --------------------------------------------------------------------
//  DnnLayerInput.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerInput_h
#define DnnLayerInput_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerInput : public DnnLayer {
    protected:
        unsigned long m_x;      // Width
        unsigned long m_y;      // Height
        unsigned long m_z;      // Depth
        unsigned long m_size;   // = x * w * h;  // Count of DNN_NUMERIC elements.

    public:
        static const char *className( void );
        
        DnnLayerInput( unsigned long xx, unsigned long yy = 1, unsigned long zz = 1 );
        virtual ~DnnLayerInput( void );
       
        virtual unsigned long aX( void ) const; // Activation dimensions
        virtual unsigned long aY( void ) const;
        virtual unsigned long aZ( void ) const;
        virtual unsigned long aSize( void ) const;

        virtual void initialize( void );                // Zero activations, gradiant and randomize weights.
        virtual void randomize(  void );                // Randomize weights and bias.

        virtual bool forward( const Matrix &data );     // Forward propagate while training
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting

    };
    

}   // namespace tfs

#endif /* DnnLayerInput_hpp */
