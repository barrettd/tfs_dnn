// --------------------------------------------------------------------
//  DnnLayerRectifiedLinearUnit.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerRectifiedLinearUnit_h
#define DnnLayerRectifiedLinearUnit_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerRectifiedLinearUnit : public DnnLayer {
    protected:
        
    public:
        static const char *className( void );

        DnnLayerRectifiedLinearUnit( DnnLayer *previousLayer );
        virtual ~DnnLayerRectifiedLinearUnit( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerRectifiedLinearUnit_h */
