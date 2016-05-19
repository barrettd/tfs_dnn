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
        
        DnnLayerInput( unsigned long xx, unsigned long yy = 1, unsigned long zz = 1, const bool trainable = true );
        virtual ~DnnLayerInput( void );
       
    };
    

}   // namespace tfs

#endif /* DnnLayerInput_hpp */
