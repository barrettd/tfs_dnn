// --------------------------------------------------------------------
//  DnnLayer.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef dnnLayer_h
#define dnnLayer_h

namespace tfs {
    
    class DnnLayer {            // Base class of all layers.
    protected:
        unsigned long m_in_x;
        unsigned long m_in_y;
        unsigned long m_in_z;      // Depth
        unsigned long m_out_x;
        unsigned long m_out_y;
        unsigned long m_out_z;      // Depth
        
    public:
        DnnLayer( void );
        DnnLayer( unsigned long xx, unsigned long yy, unsigned long zz = 1 );
        virtual ~DnnLayer( void );
        
        virtual void randomize( void ); // Randomize weights and bias.

        virtual bool forward(  void );  // Forward propagate while training
        virtual bool backprop( void );  // Back propagate while training
        virtual bool predict(  void );  // Forward progagate when predicting
        
    };
    
    
}   // namespace tfs

#endif /* dnnLayers_h */
