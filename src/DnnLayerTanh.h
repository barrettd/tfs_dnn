// --------------------------------------------------------------------
//  DnnLayerTanh.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerTanh_h
#define DnnLayerTanh_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerTanh : public DnnLayer {
    protected:
        
    public:
        DnnLayerTanh( void );
        virtual ~DnnLayerTanh( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerTanh_h */
