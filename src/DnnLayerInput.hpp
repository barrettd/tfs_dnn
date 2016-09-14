// --------------------------------------------------------------------
//  DnnLayerInput.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerInput_hpp
#define DnnLayerInput_hpp

#include "DnnLayer.hpp"

namespace tfs {
    
    class DnnLayerInput : public DnnLayer {
    public:
        DnnLayerInput( unsigned long xx, unsigned long yy = 1, unsigned long zz = 1,
                      const bool trainable = true, const bool retain_dw = false );
        virtual ~DnnLayerInput( void );
               
    };
    

}   // namespace tfs

#endif /* DnnLayerInput_hpp */
