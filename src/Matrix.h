// --------------------------------------------------------------------
//  Matrix.hpp - 3D matrix
//
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#ifndef Matrix_h
#define Matrix_h

#include "Constants.h"

namespace tfs {     // Tree Frog Software

    class Matrix {              // Possibly use Eigen matricies: https://eigen.tuxfamily.org/
    protected:
        unsigned long m_x;      // Width
        unsigned long m_y;      // Height
        unsigned long m_z;      // Depth
        unsigned long m_size;   // = x * w * h;  // Count of DNN_NUMERIC elements.
        DNN_NUMERIC  *m_data;
        DNN_NUMERIC  *m_end;

    public:
         Matrix( const Matrix &other );     // Does not copy contents.
         Matrix( const unsigned long xx, const unsigned long yy = 1, const unsigned long zz = 1 );
        ~Matrix( void );
        
        unsigned long width(  void ) const; // x
        unsigned long height( void ) const; // y
        unsigned long depth(  void ) const; // z
        unsigned long size(   void ) const; // = x * w * h;  // Count of DNN_NUMERIC elements.
        
              DNN_NUMERIC* data( void );        // Start of data array
        const DNN_NUMERIC* end(  void ) const;  // End of data array 
        const DNN_NUMERIC* dataReadOnly( void ) const;
        
        void randomize( void );                     // Fill matrix with small, positive random values
        void zero( void );                          // Fill matrix with zeros
        
        DNN_NUMERIC max( void ) const;              // Return maximum value.
        DNN_NUMERIC dot( const Matrix &rhs ) const; // Calculate the dot product.
    };

    class DMatrix {
    protected:
        unsigned long m_x;      // Width
        unsigned long m_y;      // Height
        unsigned long m_z;      // Depth
        unsigned long m_size;   // = x * w * h;  // Count of DNN_NUMERIC elements.
        DNN_INTEGER  *m_data;
        DNN_INTEGER  *m_end;
        
    public:
        DMatrix( const DMatrix &other );     // Does not copy contents.
        DMatrix( const unsigned long xx, const unsigned long yy = 1, const unsigned long zz = 1 );
        ~DMatrix( void );
        
        unsigned long width(  void ) const; // x
        unsigned long height( void ) const; // y
        unsigned long depth(  void ) const; // z
        unsigned long size(   void ) const; // = x * w * h;  // Count of DNN_NUMERIC elements.
        
              DNN_INTEGER* data( void );        // Start of data array
        const DNN_INTEGER* end(  void ) const;  // End of data array
        const DNN_INTEGER* dataReadOnly( void ) const;
        
        void randomize( void );                     // Fill matrix with small, positive random values
        void zero( void );                          // Fill matrix with zeros
        
        DNN_INTEGER max( void ) const;              // Return maximum value.
        DNN_INTEGER dot( const DMatrix &rhs ) const; // Calculate the dot product.
    };

    
    
}   // namespace tfs

#endif /* Matrix_h */
