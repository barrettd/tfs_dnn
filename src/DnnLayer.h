// --------------------------------------------------------------------
//  DnnLayer.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef dnnLayer_h
#define dnnLayer_h

#include <vector>
#include "Constants.h"

namespace tfs {
    
    class DnnLayer {            // Base class of all layers.
    protected:
        const char   *m_name;   // Used in serialization.
        unsigned long m_in_x;
        unsigned long m_in_y;
        unsigned long m_in_z;      // Depth
        unsigned long m_out_x;
        unsigned long m_out_y;
        unsigned long m_out_z;      // Depth
        
    public:
        DnnLayer( const char *name );
        DnnLayer( const char *name, unsigned long xx, unsigned long yy, unsigned long zz = 1 );
        virtual ~DnnLayer( void );
        
        const char *getName( void ) const;
        const char *setName( const char *name );
        
        virtual void randomize( void ); // Randomize weights and bias.

        virtual bool forward( const std::vector< DNN_NUMERIC > &data,
                              const std::vector< DNN_NUMERIC > &expectation );  // Forward propagate while training
        
        virtual bool backprop( void );  // Back propagate while training
        
        virtual bool predict( const std::vector< DNN_NUMERIC > &data,
                                    std::vector< DNN_NUMERIC > &prediction );   // Forward progagate when predicting
        
    };
    
    
}   // namespace tfs

#endif /* dnnLayers_h */
