// --------------------------------------------------------------------
//  DnnLayerMaxout.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerMaxout_h
#define DnnLayerMaxout_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerMaxout : public DnnLayer {
    protected:
        
    public:
        DnnLayerMaxout( void );
        virtual ~DnnLayerMaxout( void );
        
    };
    
    
}   // namespace tfs


#endif /* DnnLayerMaxout_h */
