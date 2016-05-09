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
        
    public:
        DnnLayerInput( void );
        DnnLayerInput( unsigned long xx, unsigned long yy, unsigned long zz = 1 );
        virtual ~DnnLayerInput( void );
        
    };
    

}   // namespace tfs

#endif /* DnnLayerInput_hpp */
