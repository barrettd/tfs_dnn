// --------------------------------------------------------------------
//  DnnLayerMaxout.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerMaxout_hpp
#define DnnLayerMaxout_hpp

#include "DnnLayer.hpp"

namespace tfs {
    
    class DnnLayerMaxout : public DnnLayer {
    protected:
    public:
        DnnLayerMaxout( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerMaxout( void );
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
        
    };
    
    
}   // namespace tfs


#endif /* DnnLayerMaxout_hpp */
