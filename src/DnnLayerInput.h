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
    public:
        static const char *className( void );
        
        DnnLayerInput( unsigned long xx, unsigned long yy = 1, unsigned long zz = 1 );
        virtual ~DnnLayerInput( void );
       
        virtual void randomize( void );     // Randomize weights and bias.

        virtual bool forward( const Matrix &data );   // Forward propagate while training
        virtual bool predict( const Matrix &data );   // Forward progagate when predicting

    };
    

}   // namespace tfs

#endif /* DnnLayerInput_hpp */
